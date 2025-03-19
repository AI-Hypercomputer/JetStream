# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestrates the engines with performance optimization for inference.

1. A client sends a DecodeRequest via gRPC to the server, an 'LLMOrchestrator'.
2. This gets wrapped as an 'ActiveRequest' inside the orchestrator, with a
    'return_channel' queue as a place that output tokens can be placed.
    - The ActiveRequest is placed on the 'prefill_queue'.
    - A while loop runs continuously, yielding any tokens placed on the return
      channel until an end condition is met (EOS token or max tokens).
3. There is a prefill_thread per prefill_engine, each of which runs on a
    distinct prefill_slice.
4. There is a generate_thread per generate_engine, each of which runs on a
    distinct generate_slice.
5. Within a prefill thread:
    - It attempts to pop ActiveRequests off the prefill_queue.
    - It tokenizes the request.
    - When successful, it performs a prefill operation, transfers the kv cache
      to the generation slice and pops this information (still wrapped in the
      same ActiveRequest) onto the generation queue.
6. Within a generation thread:
   - There is a queue of integers representing 'available slots'.
   - It checks if there is something on both the slots_queue and generation_
     queue.
   - If so, the kv_cache associated with that request into the decoding state
    of the generation loop at the relevant slot.
   - Regardless, it performs a step.
  - It takes the sampled tokens, and places them on a 'detokenizing_queue'.
7. Within the detokenizing thread:
  - Tokens are detokenized for every 'slot' in a given set of sampled tokens.
  - When an end condition is met, the 'slot' integer is returned to the
    respective generation queue.
  - This does mean that a single generation step may run after detokenizing
    indicates that row is no longer valid (if the detokenizing is running behind
    generation steps), this is fine as it avoids detokenizing being blocking of
    the generate thread.

If you haven't worked with concurrency in python before - queues are thread-safe
by default, so we can happily use them to transfer pointers to data between
different processes. The structure of this server is simple as a result - a
thread for each thing we might want to do (prefill, transfer, generate,
detokenize), and corresponding queues that an active request is passed between.
The same goes for the 'return_channel' of the request itself, where we can just
pop tokens once they are done and try to pop them back to transmit them over
grpc.
It is literally queues all the way down! :)
The primary concern is GIL contention between threads, which is why we block
on queues that don't have an ongoing activity (i.e. everything but the
generation queue) because we don't control to go back to those queues until
necessary. Blocking means that the GIL doesn't switch back to that thread,
wheras continual queue get operations 'chop' control and mean that we do not
achieve good throughput. This is okay on the prefill/transfer/detokenization
threads because we don't need to do anything other than react to the presence
of items on these queues, wheras the generation thread needs to also run a
step - so it cannot block until it has new things to insert.

## Testing
This server is intended to be easy to locally test.

Either use :orchestrator test, which tests the multi-threading components,
:server_test, which extends this to test grpc_components, or run it locally
to debug hangs due to bugs in threads (it is easier to debug with live logs).
"""

from datetime import datetime
import dataclasses
import functools
import itertools
import logging
import os
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from typing import Any, AsyncIterator, Optional, Tuple, cast, List

import grpc
import jax
import jax.numpy as jnp

from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.core.utils import async_multifuture
from jetstream.core.utils.return_sample import ReturnSample
from jetstream.engine import engine_api, tokenizer_api, token_utils
from jetstream.core.metrics.prometheus import JetstreamMetricsCollector
import numpy as np

log_level = os.getenv("LOG_LEVEL", "WARNING").upper()

logger = logging.getLogger("JetstreamLogger")
logger.propagate = False
logger.setLevel(getattr(logging, log_level, logging.WARNING))

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(getattr(logging, log_level, logging.WARNING))
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def ThreadDebugLog(thread_name: str, message: str) -> None:
  logger.debug("[%s] %s", thread_name, message)


@dataclasses.dataclass
class ActiveRequestMetadata:
  """Inference request metadata."""

  start_time: float = 0.0

  prefill_enqueue_time: float = 0.0
  prefill_dequeue_time: float = 0.0

  transfer_enqueue_time: float = 0.0
  transfer_dequeue_time: float = 0.0

  generate_enqueue_time: float = 0.0
  generate_dequeue_time: float = 0.0

  complete_time: float = 0.0

  def stats(self) -> str:
    return (
        f"{self.prefill_enqueue_time - self.start_time:.2f};"
        f"{self.prefill_dequeue_time - self.prefill_enqueue_time:.2f};"
        f"{time.perf_counter() - self.prefill_dequeue_time:.2f}"
    )


@dataclasses.dataclass
class ActiveRequest:
  """Current state of the driver."""

  #################### Information relevant for generation #####################
  max_tokens: int
  # We keep prefill and decode information together in the same object so that
  # there is less indirection about where this return channel is.
  # The return channel returns a list of strings, one per sample for that query.
  return_channel: async_multifuture.AsyncMultifuture[list[ReturnSample]]
  # [num_samples,] which corresponds to whether each sample is complete for the
  # requests.
  complete: Optional[np.ndarray] = None
  prefill_result: Any = None
  # The number of responses for one request.
  num_samples: int = 1
  # The unique id for the activeRequest, used for tracking the request's status
  # TODO(wyzhang): Figure out how to set request uuid without potentially
  #                causing jax.jit re-compilation for engine api implementation.
  request_id: Optional[uuid.UUID] = None
  #################### Information relevant for prefill ########################
  prefill_content: Optional[str | list[int]] = None
  ################## Information relevant for detokenization ###################
  # Which generate step this was added at.
  generate_timestep_added: Optional[int] = None
  is_client_side_tokenization: Optional[bool] = False
  ################## Information relevant for metrics ###################
  metadata: ActiveRequestMetadata = dataclasses.field(
      default_factory=ActiveRequestMetadata
  )

  def enqueue_samples(self, generated_samples: list[ReturnSample]):
    """Adds the generated sample(s) to return channel for current step.

    Args:
      generated_samples: The generated sample(s) for current step.

    This should be called only from within the Drivers background thread.
    """
    self.return_channel.add_result(generated_samples)


class JetThread(threading.Thread):
  """Thread that kills the program if it fails.

  If a driver thread goes down, we can't operate.
  """

  def run(self):
    try:
      super().run()
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Thread {self.name} encountered an error: {e}")
      traceback.print_exc()
      os.kill(os.getpid(), signal.SIGKILL)


async def AbortOrRaise(
    context: grpc.aio.ServicerContext | None,
    code: grpc.StatusCode,
    details: str,
):
  """Safely aborts a gRPC context if available, or raises an Exception."""
  if context is None:
    raise RuntimeError(details)

  await context.abort(code, details)


class Driver:
  """Drives the engines."""

  _prefill_engines: list[engine_api.Engine]
  _generate_engines: list[engine_api.Engine]
  # Allows us to pre-load the params, primarily so that we can iterate quickly
  # on the driver in colab without reloading weights.
  _prefill_params: list[Any]
  _generate_params: list[Any]
  # Stage 1
  _prefill_backlog: queue.Queue[ActiveRequest | None]
  # Stage 2
  _transfer_backlogs: list[queue.Queue[ActiveRequest]] = []
  # Stage 3
  # We keep this as a dict to avoid a possibly expensive object comparison
  # when logging the index of the generate engine we send a prefill result
  # to, it allows us to natively have the index from the min operation, rather
  # than have to call .index()
  _generate_backlogs: dict[int, queue.Queue[ActiveRequest]] = {}
  # Stage 4
  # This can be a list because we can pass it as an arg to generate and
  # detokenize threads. It is a list of tokens to be detokenized.
  _detokenize_backlogs: list[queue.Queue[engine_api.ResultTokens]] = []
  _generate_slots: list[queue.Queue[int]] = []
  _active_requests: list[queue.Queue[tuple[int, ActiveRequest]]] = []

  # For interleaved_mode, only generate if all slots are full
  # or corresponding prefill queue is empty.
  _interleaved_mode: bool = False

  # todo: remove jax_padding after all then engine migrate to np padding
  _jax_padding = True

  # If True, multiple responses will be generated for one prompt. The number
  # of responses should be specified in the request.
  _multi_sampling = False

  # All metrics we want to monitor should be collected with this
  _metrics_collector: JetstreamMetricsCollector | None = None

  def __init__(
      self,
      prefill_engines: Optional[list[engine_api.Engine]] = None,
      generate_engines: Optional[list[engine_api.Engine]] = None,
      prefill_params: Optional[list[Any]] = None,
      generate_params: Optional[list[Any]] = None,
      interleaved_mode: bool = False,
      jax_padding: bool = True,
      metrics_collector: JetstreamMetricsCollector | None = None,
      is_ray_backend: bool = False,
      multi_sampling: bool = False,
  ):
    if prefill_engines is None:
      raise ValueError("No prefill engine provided.")
    if generate_engines is None:
      raise ValueError("No generate engine provided.")
    if prefill_params is None:
      raise ValueError("No prefill parameter provided.")
    if generate_params is None:
      raise ValueError("No generate parameter provided.")

    logger.info(
        "Initializing the driver with %d prefill engines and %d "
        "generate engines in %s mode",
        len(prefill_engines),
        len(generate_engines),
        "interleaved" if interleaved_mode else "disaggregated",
    )

    self._prefill_engines = prefill_engines
    self._generate_engines = generate_engines
    self._prefill_params = prefill_params
    self._generate_params = generate_params
    self._interleaved_mode = interleaved_mode
    self._metrics_collector = metrics_collector
    self._multi_sampling = multi_sampling

    # Stages 1-4 represent the life cycle of a request.
    # Stage 1
    # At first, a request is placed here in order to get prefilled.
    self._prefill_backlog = queue.Queue()
    if self._metrics_collector:
      self._metrics_collector.get_prefill_backlog_metric().set_function(
          lambda: float(self._prefill_backlog.qsize())
      )

    # Stage 2
    # After prefilling, it is placed here in order to get transferred to
    # one of the generate backlogs.
    # Interleaved Mode: Max size is 1 to increase the HBM utilization
    # during generate.
    # Disaggregated Mode: Max size is 4 to allow for 2 prefills to be enqueued
    # while 1 transfer is enqueued while 1 is being transferred.
    # TODO: Make queue size configurable.
    self._transfer_backlogs = [
        queue.Queue(1 if self._interleaved_mode else 4)
        for i in range(len(self._prefill_engines))
    ]
    if self._metrics_collector:
      for idx, backlog in enumerate(self._transfer_backlogs):
        self._metrics_collector.get_transfer_backlog_metric(idx).set_function(
            functools.partial(float, backlog.qsize())
        )
    # Stage 3
    # Each generate engine accesses its own generate backlog.
    # Interleaved Mode: Max size is 1 to increase the HBM utilization
    # during generate.
    # Disaggregated Mode: Set as 1/3 the number of concurrent decodes.
    # TODO: Calculate the backlog to saturate the generate engine while
    # minimizing the memory usage for disaggregated mode.
    # TODO: Make queue size configurable.
    self._generate_backlogs = {
        idx: queue.Queue(
            1 if self._interleaved_mode else engine.max_concurrent_decodes // 3
        )
        for idx, engine in enumerate(self._generate_engines)
    }
    if self._metrics_collector:
      for idx, backlog in self._generate_backlogs.items():
        self._metrics_collector.get_generate_backlog_metric(idx).set_function(
            functools.partial(float, backlog.qsize())
        )
    # Stage 4
    # After generation, ActiveRequests are placed on the detokenization backlog
    # for tokens to be sent into each ActiveRequest's return channel.
    # We have one of these per generate engine to simplify the logic keeping
    # track of which generation engine to replace slots on.
    # This is a queue of either - tuple[int, ActiveRequest] which represents our
    # active requests, or tuple[int, sample_tokens]. We combine these into one
    # queue because it allows us to be somewhat clever with how we do
    # detokenization.
    # If the detokenization receives an (int, ActiveRequest) this signifies
    # that slot int should from now be placing tokens in the return channel of
    # the ActiveRequest.
    # If it receives (int, sample_tokens) then it actually
    # does a detokenization for any slots which have previously been set active
    # via the previous kind of object, and the int is used to log which step
    # the tokens were created at. By having them in one queue we prevent
    # the possibility of race conditions where a slot is made live before the
    # tokens are ready and it receives tokens from a different sequence,
    # or tokens detokenized before the relevant slot is live.
    self._detokenize_backlogs = [
        # We don't let detokenization accumulate more than 8 steps to avoid
        # synchronization issues.
        queue.Queue(8)
        for _ in self._generate_engines
    ]

    # A queue of integers representing available 'slots' in the decode
    # operation. I.e. potentially available rows in the batch and/or microbatch.
    # When we want to insert a prefill result, we pop an integer to insert at.
    # When this is empty, it means all slots are full.
    self._generate_slots = [
        queue.Queue(engine.max_concurrent_decodes)
        for engine in self._generate_engines
    ]
    _ = [
        [
            self._generate_slots[idx].put(i)
            for i in range(engine.max_concurrent_decodes)
        ]
        for idx, engine in enumerate(self._generate_engines)
    ]

    logger.debug(
        "Initializing the driver with 1 prefill backlogs, "
        "%d transfer backlogs, \n"
        "%d generate backlogs and %d detokenize backlogs.",
        len(self._transfer_backlogs),
        len(self._generate_backlogs),
        len(self._detokenize_backlogs),
    )

    self._jax_padding = jax_padding

    # Create all threads
    self._prefill_threads = [
        JetThread(
            target=functools.partial(self._prefill_thread, idx),
            name=f"prefill-{idx}",
            daemon=True,
        )
        for idx in range(len(self._prefill_engines))
    ]
    self._transfer_threads = [
        JetThread(
            target=functools.partial(
                self._transfer_thread,
                idx,
            ),
            name=f"transfer-{idx}",
            daemon=True,
        )
        for idx in range(len(self._prefill_engines))
    ]
    self._generate_threads = [
        JetThread(
            target=functools.partial(
                self._generate_thread,
                idx,
            ),
            name=f"generate-{idx}",
            daemon=True,
        )
        for idx in range(len(self._generate_engines))
    ]
    self.detokenize_threads = [
        JetThread(
            target=functools.partial(
                self._detokenize_thread,
                idx,
            ),
            name=f"detokenize-{idx}",
        )
        for idx in range(len(self._generate_engines))
    ]
    self._all_threads = list(
        itertools.chain(
            self._prefill_threads,
            self._transfer_threads,
            self._generate_threads,
            self.detokenize_threads,
        )
    )
    self.live = True
    self._is_ray_backend = is_ray_backend
    # Start all threads
    for t in self._all_threads:
      t.start()

    logger.debug(
        "Started %d prefill threads, %d transfer threads, \n"
        "%d generate threads, and %d detokenize threads.",
        len(self._prefill_threads),
        len(self._transfer_threads),
        len(self._generate_threads),
        len(self.detokenize_threads),
    )

    logger.info("Driver initialized.")

  def stop(self):
    """Stops the driver and all background threads."""
    logger.info("Stopping the driver and all background threads...")
    # Signal to all threads that they should stop.
    self.live = False

    all_backlogs = list(
        itertools.chain(
            [self._prefill_backlog],
            self._transfer_backlogs,
            self._generate_backlogs.values(),
            self._detokenize_backlogs,
        )
    )

    while any(t.is_alive() for t in self._all_threads):
      # Empty all backlogs and mark any remaining requests as cancelled.
      for q in all_backlogs:
        while True:
          try:
            r = q.get_nowait()
            if r is None:
              continue
            elif isinstance(r, ActiveRequest):
              r.return_channel = None
            else:  # detokenize backlog
              _, r = r
              if isinstance(r, ActiveRequest):
                r.return_channel = None
          except queue.Empty:
            break

      # Put sentinels to unblock threads.
      for q in all_backlogs:
        try:
          q.put_nowait(None)
        except queue.Full:
          pass

    # Wait for all threads to stop.
    for t in self._all_threads:
      t.join()

    logger.info("Driver stopped.")

  def get_total_concurrent_requests(self) -> int:
    """Gets the total number of concurrent requests the driver can handle."""
    # We don't support filling all backlogs at once because it can cause GIL
    # contention.
    total_max_concurrent_decodes = sum(
        [e.max_concurrent_decodes for e in self._generate_engines]
    )
    return total_max_concurrent_decodes

  def prefill_backlog_size(self):
    return self._prefill_backlog.qsize()

  def place_request_on_prefill_queue(self, request: ActiveRequest):
    """Used to place new requests for prefilling and generation."""
    # Don't block so we can fail and shed load when the queue is full.
    self._prefill_backlog.put(request, block=False)

  def _process_prefill_content(
      self,
      request: ActiveRequest,
      tokenizer: tokenizer_api.Tokenizer,
      is_bos: bool,
      max_prefill_length: int,
      chunked_prefill: bool = False,
      chunk_size: Optional[int] = None,
  ) -> Tuple[jax.Array | np.ndarray, jax.Array, jax.Array | np.ndarray]:
    content = request.prefill_content
    if isinstance(content, str):
      # If it's text input, tokenize and pad the input.
      tokens, true_length = tokenizer.encode(
          content,
          is_bos=is_bos,
          max_prefill_length=max_prefill_length,
          jax_padding=self._jax_padding,
      )
      positions = jnp.expand_dims(
          jnp.arange(0, len(tokens), dtype=jnp.int32), 0
      )

      if chunked_prefill:
        return token_utils.chunk_and_pad_tokens(
            tokens[:true_length],
            tokenizer.bos_id,
            tokenizer.pad_id,
            is_bos=is_bos,
            max_prefill_length=max_prefill_length,
            chunk_size=chunk_size,
            jax_padding=self._jax_padding,
        )
      return tokens, true_length, positions

    else:
      if chunked_prefill:
        return token_utils.chunk_and_pad_tokens(
            content,
            tokenizer.bos_id,
            tokenizer.pad_id,
            is_bos=is_bos,
            max_prefill_length=max_prefill_length,
            chunk_size=chunk_size,
            jax_padding=self._jax_padding,
        )

      # If it's token input, pad the input.
      tokens, true_length = token_utils.pad_tokens(
          content,
          tokenizer.bos_id,
          tokenizer.pad_id,
          is_bos=is_bos,
          max_prefill_length=max_prefill_length,
          jax_padding=self._jax_padding,
      )
      positions = jnp.expand_dims(
          jnp.arange(0, len(tokens), dtype=jnp.int32), 0
      )
      return tokens, true_length, positions

  def _prefill_thread(self, idx: int):
    """Thread which runs in the background performing prefills."""
    logger.info("Spinning up prefill thread %d.", idx)
    prefill_engine = self._prefill_engines[idx]
    prefill_params = self._prefill_params[idx]
    metadata = prefill_engine.get_tokenizer()
    tokenizer = prefill_engine.build_tokenizer(metadata)
    thread_name = f"Prefill thread {idx}"
    ThreadDebugLog(thread_name, f"Prefill params {idx} loaded.")

    while self.live:
      my_transfer_backlog = self._transfer_backlogs[idx]
      # The prefill thread can just sleep until it has work to do.
      request = self._prefill_backlog.get(block=True)

      if request is None:
        break
      request.metadata.prefill_dequeue_time = time.perf_counter()
      is_bos = True
      ThreadDebugLog(
          thread_name,
          f"Executing prefilling for one ActiveRequest. Current prefill "
          f"backlog size: {self._prefill_backlog.qsize()},"
          f" is_bos: {is_bos}",
      )
      # Tokenize and padding the text or token input.
      padded_tokens, true_length, _ = self._process_prefill_content(
          request,
          tokenizer,
          is_bos,
          prefill_engine.max_prefill_length,
          False,
      )

      # Compute new kv cache for the prefill_content.
      if self._multi_sampling:
        prefill_result, first_token = prefill_engine.prefill_multisampling(
            params=prefill_params,
            padded_tokens=padded_tokens,
            true_length=true_length,
            num_samples=request.num_samples,
        )
        request.complete = np.zeros((request.num_samples,), np.bool_)
      else:
        # if chunked_prefill is used,
        if prefill_engine.use_chunked_prefill:
          padded_chunked_tokens, true_lengths_of_chunks, positions_chunks = (
              self._process_prefill_content(
                  request,
                  tokenizer,
                  is_bos,
                  prefill_engine.max_prefill_length,
                  prefill_engine.use_chunked_prefill,
                  prefill_engine.prefill_chunk_size,
              )
          )
          prefill_result = None
          for chunk_num, _ in enumerate(padded_chunked_tokens):
            cache_so_far = (
                {} if prefill_result is None else prefill_result["cache"]  # pylint: disable=unsubscriptable-object
            )
            prefill_result, first_token = prefill_engine.prefill(
                params=prefill_params | {"cache": cache_so_far},
                padded_tokens=padded_chunked_tokens[chunk_num],
                true_length=true_lengths_of_chunks[chunk_num],
                positions=positions_chunks[chunk_num],
                previous_chunk=prefill_result,
                complete_prompt_true_length=true_length,
                complete_padded_prompt=padded_tokens,
            )
            # true_length_array is arrays of 1 true lengths so far
            t_l_array = jnp.expand_dims(
                jnp.arange(
                    0,
                    chunk_num * prefill_engine.prefill_chunk_size
                    + true_lengths_of_chunks[chunk_num],
                ),
                1,
            )
            prefill_result["true_length_array"] = t_l_array
        else:
          # Compute new kv cache for the prefill_content.
          prefill_result, first_token = prefill_engine.prefill(
              params=prefill_params,
              padded_tokens=padded_tokens,
              true_length=true_length,
          )
      request.prefill_result = prefill_result
      request.complete = np.zeros((prefill_engine.samples_per_slot,), np.bool_)

      # put first token to detokenize queue
      my_detokenize_backlog = self._detokenize_backlogs[idx]
      request.metadata.transfer_enqueue_time = time.perf_counter()
      my_detokenize_backlog.put(
          (first_token, request, request.metadata.prefill_dequeue_time),
          block=True,
      )

      ThreadDebugLog(thread_name, "Completed prefilling for one ActiveRequest.")
      # Once prefill is complete, place it on the transfer queue and block if
      # full.
      my_transfer_backlog.put(request, block=True)
      ThreadDebugLog(
          thread_name,
          f"Placed request on transfer backlog {idx}. "
          f"Current transfer backlog size: {my_transfer_backlog.qsize()}.",
      )
      if self._metrics_collector:
        self._metrics_collector.get_request_input_length().observe(true_length)

      if self._metrics_collector:
        self._metrics_collector.get_time_per_prefill_token().observe(
            (
                request.metadata.transfer_enqueue_time
                - request.metadata.prefill_dequeue_time
            )
            / true_length
        )

      del prefill_result
      del request

    logger.info("Prefill thread %d stopped.", idx)

  def _jax_transfer_prefill_result(
      self, new_request: ActiveRequest, target_idx: int
  ):
    new_request.prefill_result = jax.device_put(
        new_request.prefill_result,
        self._generate_engines[target_idx].get_prefix_destination_sharding(),
    )
    # Block here so we don't block on the generate thread that steps.
    jax.block_until_ready(new_request.prefill_result)

  def _ray_transfer_prefill_result(
      self, new_request: ActiveRequest, target_idx: int
  ):
    self._generate_engines[target_idx].transfer(new_request.prefill_result)

  def _transfer_prefill_result(
      self, new_request: ActiveRequest, target_idx: int
  ):
    if self._is_ray_backend:
      self._ray_transfer_prefill_result(new_request, target_idx)
    else:
      self._jax_transfer_prefill_result(new_request, target_idx)

  def _transfer_thread(self, idx: int):
    """Transfers the kv cache on an active request to the least full
    generate backlog."""
    logger.info("Spinning up transfer thread %d.", idx)
    thread_name = f"Transfer thread {idx}"
    transfer_backlog = self._transfer_backlogs[idx]

    while self.live:
      # The transfer thread can just sleep until it has work to do.
      new_request = transfer_backlog.get(block=True)
      if new_request is None:
        break
      new_request.metadata.transfer_dequeue_time = time.perf_counter()
      target_idx = min(
          self._generate_backlogs.items(), key=lambda q: q[1].qsize()
      )[0]
      # Only transfer the KVCache for the disaggregated serving.
      # TODO: Remove the conditional after fixing the compatibility.
      if not self._interleaved_mode:
        ThreadDebugLog(
            thread_name,
            f"Transferring prefill result from prefill engine {idx} "
            f"to generate engine {target_idx}.",
        )
        # Transfer the info to the relevant generate slice.
        self._transfer_prefill_result(new_request, target_idx)
      # Place the request on the correct generate backlog and block if full.
      new_request.metadata.generate_enqueue_time = time.perf_counter()
      self._generate_backlogs[target_idx].put(new_request, block=True)
      ThreadDebugLog(
          thread_name,
          f"Transferred ActiveRequest from prefill engine {idx} "
          f"to generate backlog {target_idx}. "
          f"Current generate backlog size: "
          f"{self._generate_backlogs[target_idx].qsize()}.",
      )

    logger.info("Transfer thread %d stopped.", idx)

  def _insert_if_possible(
      self,
      idx,
      thread_name,
      max_concurrent_decodes,
      generate_timestep,
      decode_state,
      my_slots,
      my_generate_backlog,
      generate_engine,
      my_detokenize_backlog,
  ):
    # Check if there are any free my_slots. We don't want to block here since
    # we can still generate if we can't insert. We do this in a while loop to
    # insert as many sequences as possible.
    while True:
      my_slots_size = my_slots.qsize()

      try:
        slot = my_slots.get(block=False)
        # Found a slot, now see if we can fill it.
      except queue.Empty:
        # Exit this while loop as we have no free slots to insert into.
        ThreadDebugLog(thread_name, "All slots are occupied.")
        break

      ThreadDebugLog(thread_name, "Got an available slot.")

      # We block when the decode slots are all free since we need to get a
      # prefilled request to insert. We add timeout for the block to handle
      # the case when the prefill backlog is cancelled and we end up with no
      # more useful prefill work to do.
      block = my_slots_size == max_concurrent_decodes
      if self._interleaved_mode:
        # For interleaved mode, we also blocks when prefill backlog
        # is not empty or there are transfer work to do.
        block |= not self._prefill_backlog.empty()
        for transfer_backlog in self._transfer_backlogs:
          block |= not transfer_backlog.empty()
      try:
        new_request = my_generate_backlog.get(block=block, timeout=1.0)
        if new_request is None:
          return None
        ThreadDebugLog(
            thread_name,
            f"Got a new ActiveRequest from generate backlog {idx}.",
        )
        new_request.metadata.generate_dequeue_time = time.perf_counter()
        if (
            self._metrics_collector
            and new_request.metadata.start_time is not None
        ):
          self._metrics_collector.get_queue_duration().observe(
              # Time in prefill queue
              new_request.metadata.prefill_dequeue_time
              - new_request.metadata.prefill_enqueue_time
              # Time in transfer queue
              + new_request.metadata.transfer_dequeue_time
              - new_request.metadata.transfer_enqueue_time
              # Time in generate queue
              + new_request.metadata.generate_dequeue_time
              - new_request.metadata.generate_enqueue_time
          )
        # Got free slot and new request, use them.
      except queue.Empty:
        # No new requests, we can't insert, so put back slot.
        my_slots.put(slot, block=False)
        ThreadDebugLog(
            thread_name,
            f"No new ActiveRequest from generate backlog {idx}. "
            f"Put back the slot.",
        )
        # If we were blocking and hit the timeout, then retry the loop.
        # Otherwise, we can exit and proceed to generation.
        if block:
          continue
        else:
          break

      decode_state = generate_engine.insert(
          new_request.prefill_result,
          decode_state,
          slot=slot,
          request_id=new_request.request_id,
      )
      ThreadDebugLog(
          thread_name,
          f"Generate slice {idx} filled slot {slot} at step "
          f"{generate_timestep}.",
      )

      del new_request.prefill_result
      new_request.generate_timestep_added = generate_timestep
      new_request.complete = np.zeros(
          (generate_engine.samples_per_slot,), dtype=np.bool_
      )

      # Respond to detokenization backpressure.
      my_detokenize_backlog.put((slot, new_request), block=True)
      ThreadDebugLog(
          thread_name,
          f"Put the ActiveRequest into detokenize backlog {idx}. "
          f"Current detokenize backlog size: "
          f"{my_detokenize_backlog.qsize()}.",
      )
    return decode_state

  def _bulk_insert_if_possible(
      self,
      idx,
      thread_name,
      max_concurrent_decodes,
      generate_timestep,
      decode_state,
      my_slots,
      my_generate_backlog,
      generate_engine,
      my_detokenize_backlog,
  ):
    while True:
      my_slots_size = my_slots.qsize()
      # We block when the decode slots are all free since we need to get a
      # prefilled request to insert. We add timeout for the block to handle
      # the case when the prefill backlog is cancelled and we end up with no
      # more useful prefill work to do.
      block = my_slots_size == max_concurrent_decodes
      if self._interleaved_mode:
        # For interleaved mode, we also blocks when prefill backlog
        # is not empty or there are transfer work to do.
        block |= not self._prefill_backlog.empty()
        for transfer_backlog in self._transfer_backlogs:
          block |= not transfer_backlog.empty()

      if my_generate_backlog.empty():
        if block:
          continue
        else:
          break

      if my_generate_backlog.queue[0] is None:
        return None

      expected_slots = my_generate_backlog.queue[0].num_samples

      if expected_slots > max_concurrent_decodes:
        raise RuntimeError(
            f"Expect {expected_slots} slots, but there are only"
            "{max_concurrent_decodes} slots available."
        )

      available_slots = []
      try:
        for _ in range(expected_slots):
          slot = my_slots.get(block=False)
          available_slots.append(slot)
      except queue.Empty:
        # Exit this while loop as we don't have enough slots to insert into.
        ThreadDebugLog(
            thread_name, f"Not enough slots. Expected {expected_slots} slots."
        )
        for slot in available_slots:
          my_slots.put(slot, block=False)
        break

      ThreadDebugLog(thread_name, "Got enough available slots.")

      try:
        new_request = my_generate_backlog.get(block=False)
        if new_request is None:
          return None
        ThreadDebugLog(
            thread_name,
            f"Got a new ActiveRequest from generate backlog {idx}.",
        )
        new_request.metadata.generate_dequeue_time = time.perf_counter()
        if (
            self._metrics_collector
            and new_request.metadata.start_time is not None
        ):
          self._metrics_collector.get_queue_duration().observe(
              # Time in prefill queue
              new_request.metadata.prefill_dequeue_time
              - new_request.metadata.prefill_enqueue_time
              # Time in transfer queue
              + new_request.metadata.transfer_dequeue_time
              - new_request.metadata.transfer_enqueue_time
              # Time in generate queue
              + new_request.metadata.generate_dequeue_time
              - new_request.metadata.generate_enqueue_time
          )
        # Got free slot and new request, use them.
      except queue.Empty as e:
        # We should at least have one new request at this step.
        # If not, throw a runtime error.
        raise RuntimeError("Generate backlog is empty.") from e

      decode_state = generate_engine.bulk_insert(
          new_request.prefill_result, decode_state, slots=available_slots
      )

      del new_request.prefill_result
      new_request.generate_timestep_added = generate_timestep
      new_request.complete = np.zeros(
          (new_request.num_samples,), dtype=np.bool_
      )
      # Respond to detokenization backpressure.

      my_detokenize_backlog.put((available_slots, new_request), block=True)
      ThreadDebugLog(
          thread_name,
          f"Put the ActiveRequest into detokenize backlog {idx}. "
          f"Current detokenize backlog size: "
          f"{my_detokenize_backlog.qsize()}.",
      )
    return decode_state

  def _generate_thread(self, idx: int):
    """Step token generation and insert prefills from backlog."""
    logger.info("Spinning up generate thread %d.", idx)
    generate_engine = self._generate_engines[idx]
    my_slots = self._generate_slots[idx]
    my_generate_backlog = self._generate_backlogs[idx]
    my_detokenize_backlog = self._detokenize_backlogs[idx]

    # Keep track of what step tokens were generated at.
    generate_timestep = 0
    # State to store things like running kv cache in.
    decode_state = generate_engine.init_decode_state()

    generate_params = self._generate_params[idx]
    thread_name = f"Generate thread {idx}"
    ThreadDebugLog(thread_name, f"Generate params {idx} loaded.")
    time_of_last_generate = time.time()
    time_of_last_print = time.time()
    while self.live:
      if (time.time() - time_of_last_print) > 1:
        ThreadDebugLog(
            thread_name,
            f"Generate thread making a decision with:"
            f" prefill_backlog={self._prefill_backlog.qsize()}"
            f" generate_free_slots={my_slots.qsize()}",
        )
        time_of_last_print = time.time()

      max_concurrent_decodes = generate_engine.max_concurrent_decodes

      if self._metrics_collector:
        self._metrics_collector.get_slots_used_percentage_metric(
            idx
        ).set_function(
            lambda: float(1 - (my_slots.qsize() / max_concurrent_decodes))
        )

      if self._multi_sampling:
        decode_state = self._bulk_insert_if_possible(
            idx,
            thread_name,
            max_concurrent_decodes,
            generate_timestep,
            decode_state,
            my_slots,
            my_generate_backlog,
            generate_engine,
            my_detokenize_backlog,
        )
      else:
        decode_state = self._insert_if_possible(
            idx,
            thread_name,
            max_concurrent_decodes,
            generate_timestep,
            decode_state,
            my_slots,
            my_generate_backlog,
            generate_engine,
            my_detokenize_backlog,
        )
      if decode_state is None:
        break

      # At this point, we know that we have at least some slots filled.
      assert (
          my_slots.qsize() < max_concurrent_decodes
      ), "At this point we must have some requests inserted into the slots."

      # Now we actually take a generate step on requests in the slots.
      decode_state, sampled_tokens = generate_engine.generate(
          generate_params, decode_state
      )
      sampled_tokens.copy_to_host_async()
      # Respond to detokenization backpressure.
      my_detokenize_backlog.put((generate_timestep, sampled_tokens), block=True)
      generate_timestep += 1
      ThreadDebugLog(
          thread_name,
          f"Step {generate_timestep} - slots free : {my_slots.qsize()} / "
          f"{max_concurrent_decodes}, took "
          f"{((time.time() - time_of_last_generate) * 10**3):.2f}ms",
      )
      time_of_last_generate = time.time()

    logger.info("Generate thread %d stopped.", idx)

  def _collect_metrics(
      self,
      result_tokens: engine_api.ResultTokens,
      request: ActiveRequest,
      slot: int,
  ):
    self._metrics_collector.get_request_output_length().observe(
        result_tokens.get_result_at_slot(slot).lengths
    )
    self._metrics_collector.get_request_success_count_metric().inc()
    self._metrics_collector.get_time_per_output_token().observe(
        (
            request.metadata.complete_time
            - request.metadata.transfer_enqueue_time
        )
        / result_tokens.get_result_at_slot(slot).lengths
    )
    self._metrics_collector.get_time_per_request().observe(
        request.metadata.complete_time - request.metadata.transfer_enqueue_time
    )

    if request.metadata.start_time:
      total_time = request.metadata.complete_time - request.metadata.start_time
      prefill_time = (
          request.metadata.transfer_enqueue_time
          - request.metadata.prefill_dequeue_time
      )
      generate_time = (
          request.metadata.complete_time
          - request.metadata.generate_dequeue_time
      )
      self._metrics_collector.get_wait_time_per_request().observe(
          total_time - prefill_time - generate_time
      )

  def _detokenize_thread(self, idx: int):
    """Detokenize sampled tokens and returns them to the user."""
    # One of these per generate engine.
    # For all filled my_slots, pop the sampled token onto the relevant
    # requests return channel. If it done, place it back onto free slots.
    logger.info("Spinning up detokenize thread %d.", idx)
    my_detokenize_backlog = self._detokenize_backlogs[idx]
    my_generate_engine = self._generate_engines[idx]
    my_slots = self._generate_slots[idx]

    metadata = my_generate_engine.get_tokenizer()
    tokenizer = my_generate_engine.build_tokenizer(metadata)
    my_live_requests = {
        i: None for i in range(my_generate_engine.max_concurrent_decodes)
    }
    my_live_multi_sampling_requests = (
        {}
    )  # Mapping from a tuple of slots to a multi-sampling request.
    thread_name = f"Detokenize thread {idx}"
    while self.live:
      ThreadDebugLog(thread_name, "Waiting for a detokenization task.")
      data = my_detokenize_backlog.get(block=True)
      if data is None:
        break
      start_detokenize_time = time.time()
      # prefill first token
      if isinstance(data[0], engine_api.ResultTokens):
        request_first_token, request, _ = data
        request_first_token = request_first_token.convert_to_numpy()

        ThreadDebugLog(
            thread_name, "Detokenizing the first token of a sequence."
        )
        results, complete = token_utils.process_result_tokens(
            tokenizer=tokenizer,
            slots=0,
            slot_max_length=request.max_tokens,
            result_tokens=request_first_token,
            is_client_side_tokenization=request.is_client_side_tokenization,
            complete=request.complete,
        )
        request.complete = complete
        # Return some output samples.
        request.enqueue_samples(results)

        first_token_return_time = time.perf_counter()
        if self._metrics_collector:
          self._metrics_collector.get_time_to_first_token().observe(
              first_token_return_time - request.metadata.prefill_dequeue_time
          )

        ThreadDebugLog(
            thread_name,
            "TTFT duration: {ttft}ms".format(  # pylint: disable=consider-using-f-string
                ttft=(
                    first_token_return_time
                    - request.metadata.prefill_dequeue_time
                )
                * 1000
            ),
        )
      # generate step tokens
      elif isinstance(data[1], engine_api.ResultTokens):
        # We want to detokenize them.
        generate_timestep_added, result_tokens = data
        # Disable attribute error because pytype doesn't know this
        # is a result tokens, and we can't annotate the tuple.
        result_tokens = result_tokens.convert_to_numpy()

        if self._multi_sampling:
          requests_to_be_cleaned = []
          for slots, request in my_live_multi_sampling_requests.items():
            results, complete = token_utils.process_result_tokens(
                tokenizer=tokenizer,
                slots=slots,
                slot_max_length=request.max_tokens,
                result_tokens=result_tokens,
                is_client_side_tokenization=request.is_client_side_tokenization,
                complete=request.complete,
            )
            request.complete = complete
            # Return some output samples.
            request.enqueue_samples(results)
            if request.complete.all():
              request.metadata.complete_time = time.perf_counter()
              request.return_channel.close()
              if self._metrics_collector:
                self._collect_metrics(result_tokens, request, slots[0])
              requests_to_be_cleaned.append(slots)
              # Place the slot back on the free queue.
              for slot in slots:
                my_slots.put(
                    slot, block=False
                )  # This should always have space.
                my_generate_engine.free_resource(slot)
          for slots in requests_to_be_cleaned:
            del my_live_multi_sampling_requests[slots]
        else:
          for slot, request in my_live_requests.items():
            if request is not None:
              results, complete = token_utils.process_result_tokens(
                  tokenizer=tokenizer,
                  slots=slot,
                  slot_max_length=request.max_tokens,
                  result_tokens=result_tokens,
                  is_client_side_tokenization=(
                      request.is_client_side_tokenization
                  ),
                  complete=request.complete,
              )
              request.complete = complete
              # Return some output samples.
              request.enqueue_samples(results)
              if request.complete.all():
                request.metadata.complete_time = time.perf_counter()
                request.return_channel.close()
                if self._metrics_collector:
                  self._collect_metrics(result_tokens, request, slot)
                # Place the slot back on the free queue.
                my_live_requests[slot] = None
                my_slots.put(
                    slot, block=False
                )  # This should always have space.
                my_generate_engine.free_resource(slot)

        ThreadDebugLog(
            thread_name,
            f"Detokenizing generate step {generate_timestep_added} "
            f"took {((time.time() - start_detokenize_time) * 10**3): .2f}ms",
        )
      else:
        if self._multi_sampling:
          slots, active_request = data
          my_live_multi_sampling_requests[tuple(slots)] = active_request
        else:
          # We want to update a slot with the new channel.
          slot, active_request = data
          my_live_requests[slot] = active_request

    logger.info("Detokenize thread %d stopped.", idx)


class LLMOrchestrator(jetstream_pb2_grpc.OrchestratorServicer):
  """Coordinates a set of prefill and generate slices for LLM decoding."""

  _driver: Driver

  def __init__(self, driver: Driver):
    self._driver = driver

  def _get_prefill_content(
      self, request: jetstream_pb2.DecodeRequest
  ) -> Tuple[str | list[int], bool]:
    which_content = request.WhichOneof("content")
    content = getattr(request, which_content)
    if which_content == "text_content":
      return cast(jetstream_pb2.DecodeRequest.TextContent, content).text, False
    else:
      return (
          list(
              cast(jetstream_pb2.DecodeRequest.TokenContent, content).token_ids
          ),
          True,
      )

  def _process_client_side_tokenization_response(
      self, response: list[ReturnSample]
  ):
    samples = []
    for sample in response:
      samples.append(
          jetstream_pb2.DecodeResponse.StreamContent.Sample(
              token_ids=sample.token_ids,
          )
      )
    return jetstream_pb2.DecodeResponse(
        stream_content=jetstream_pb2.DecodeResponse.StreamContent(
            samples=samples
        )
    )

  def should_buffer_response(self, response: List[ReturnSample]) -> bool:
    for item in response:
      if item.text and token_utils.is_byte_token(item.text[-1]):
        # If any sample ends in bytes, this means we might still need to
        # decode more bytes to compose the string.
        return True

  def _process_server_side_tokenization_response(
      self, response: list[ReturnSample], buffered_response_list
  ):
    # Flush the buffered responses to each sample of current response.
    current_response_with_flushed_buffer = list(
        zip(*buffered_response_list, response)
    )
    # Empty buffer: [[s0_cur], [s1_cur], ...]
    # Has buffer:
    # [[s0_b0, s0_b1, ..., s0_cur], [s1_b0, s1_b1, ..., s1_cur], ...]
    current_response_with_flushed_buffer = cast(
        list[list[ReturnSample]], current_response_with_flushed_buffer
    )
    # Form correct sample(s) and return as StreamContent for this iteration.
    samples = []
    for sample in current_response_with_flushed_buffer:
      text = []
      token_ids = []
      for resp in sample:
        text.extend(resp.text)
        token_ids.extend(resp.token_ids)
      samples.append(
          jetstream_pb2.DecodeResponse.StreamContent.Sample(
              text=token_utils.text_tokens_to_str(text),
              token_ids=token_ids,
          )
      )
    return jetstream_pb2.DecodeResponse(
        stream_content=jetstream_pb2.DecodeResponse.StreamContent(
            samples=samples
        )
    )

  async def Decode(  # pylint: disable=invalid-overridden-method
      self,
      request: jetstream_pb2.DecodeRequest,
      context: Optional[grpc.aio.ServicerContext] = None,
  ) -> AsyncIterator[jetstream_pb2.DecodeResponse]:
    """Decode."""
    request_start_time = time.perf_counter()
    ttft = 0
    if context is None:
      logger.warning(
          "LLM orchestrator is being used in offline test mode, and will not"
          " respond to gRPC queries - only direct function calls."
      )
    is_client_side_tokenization = False
    return_channel = async_multifuture.AsyncMultifuture()
    if context:
      context.add_done_callback(return_channel.cancel)
    prefill_content, is_client_side_tokenization = self._get_prefill_content(
        request
    )
    # Wrap request as an ActiveRequest.
    active_request = ActiveRequest(
        max_tokens=request.max_tokens,
        prefill_content=prefill_content,
        is_client_side_tokenization=is_client_side_tokenization,
        return_channel=return_channel,
        metadata=ActiveRequestMetadata(
            start_time=request.metadata.start_time,
            prefill_enqueue_time=time.perf_counter(),
        ),
        num_samples=request.num_samples if request.num_samples else 1,
    )
    # The first stage is being prefilled, all other stages are handled
    # inside the driver (transfer, generate*N, detokenize).
    try:
      self._driver.place_request_on_prefill_queue(active_request)
    except queue.Full:
      # Safely abort the gRPC server thread with a retriable error.
      await AbortOrRaise(
          context=context,
          code=grpc.StatusCode.RESOURCE_EXHAUSTED,
          details=(
              "The driver prefill queue is full and more requests cannot be"
              " handled. You may retry this request."
          ),
      )
    logger.info(
        "Placed request on the prefill queue.",
    )
    # When an active request is created a queue is instantiated. New tokens
    # are placed there during the decoding loop, we pop from that queue by
    # using the .next method on the active request.
    # Yielding allows for the response to be a streaming grpc call - which
    # can be called via iterating over a for loop on the client side.
    # The DecodeResponse stream should consume all generated tokens in
    # return_channel when complete signal is received (AsyncMultifuture
    # promises this).
    buffered_response_list = []
    async for response in active_request.return_channel:
      response = cast(list[ReturnSample], response)
      if ttft == 0:
        ttft = time.perf_counter() - request_start_time
        if ttft > 2.0:
          logger.info(  # pylint: disable=logging-fstring-interpolation
              f"{datetime.now()}: "
              f"Slow TTFT: {ttft:.2f}s,"
              f" stats={active_request.metadata.stats()},"
              f" prefill_qsize={self._driver.prefill_backlog_size()}",
          )
      if is_client_side_tokenization:
        # If is_client_side_tokenization, the client should request with token
        # ids, and the JetStream server will return token ids as response.
        # The client should take care of tokenization and detokenization.
        yield self._process_client_side_tokenization_response(response)
      else:
        # Buffer response mechanism is used to handle streaming
        # detokenization with special character (For some edge cases with
        # SentencePiece tokenizer, it requires to decode a complete sequence
        # instead of a single token).
        if self.should_buffer_response(response):
          buffered_response_list.append(response)
          continue
        yield self._process_server_side_tokenization_response(
            response, buffered_response_list
        )
        # Reset buffer after flushed.
        buffered_response_list = []

  async def HealthCheck(  # pylint: disable=invalid-overridden-method
      self,
      request: jetstream_pb2.HealthCheckRequest,
      context: Optional[grpc.aio.ServicerContext] = None,
  ) -> jetstream_pb2.HealthCheckResponse:
    """HealthCheck."""
    if context is None:
      logger.warning(
          "LLM orchestrator is being used in offline test mode, and will not"
          " respond to gRPC queries - only direct function calls."
      )
    is_live = self._driver.live
    return jetstream_pb2.HealthCheckResponse(is_live=is_live)
