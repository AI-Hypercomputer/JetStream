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

"""Orchestrates the engines for the inference workflow with performance optimization.

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
achieve good throughput. This is okay on the prefill/transfer/detokenisation
threads because we don't need to do anything other than react to the presence
of items on these queues, wheras the generation thread needs to also run a
step - so it cannot block until it has new things to insert.

## Testing
This server is intended to be easy to locally test.

Either use :orchestrator test, which tests the multi-threading components,
:server_test, which extends this to test grpc_components, or run it locally
to debug hangs due to bugs in threads (it is easier to debug with live logs).
"""

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
from typing import Any, Iterable, Optional, Union

import grpc
import jax
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.engine import engine_api
from jetstream.engine import token_utils
import numpy as np


root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
root.addHandler(handler)


def delete_pytree(p):
  def delete_leaf(leaf):
    if isinstance(leaf, jax.Array):
      leaf.delete()
    del leaf

  jax.tree_map(delete_leaf, p)


@dataclasses.dataclass
class ActiveRequest:
  """Current state of the driver."""

  #################### Information relevant for generation #####################
  max_tokens: int
  # [num_samples,] which corresponds to whether each sample is complete for the
  # requests.
  complete: Optional[np.ndarray] = None
  prefill_result: Any = None
  #################### Information relevant for prefill ########################
  history_path: Optional[str] = None
  prefill_text: Optional[str] = None
  ################## Information relevant for detokenisation ###################
  # Which generate step this was added at.
  generate_timestep_added: Optional[int] = None
  # We keep prefill and decode information together in the same object so that
  # there is less indirection about where this return channel is.
  # The return channel returns a list of strings, one per sample for that query.
  return_channel: queue.Queue[list[str]] = dataclasses.field(
      default_factory=queue.Queue
  )

  def next(self: 'ActiveRequest'):
    """Blocks until the next token is available, call from RPC threads."""
    return self.return_channel.get()

  def enqueue_tokens(self, generated_tokens: list[str]):
    """Records information about the step.

    Args:
      generated_tokens: One token to put into the return channel

    This should be called only from within the Drivers background thread.
    """
    self.return_channel.put(generated_tokens)


class JetThread(threading.Thread):
  """Thread that kills the program if it fails.

  If a driver thread goes down, we can't operate.
  """

  def run(self):
    try:
      super().run()
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Thread {self.name} encountered an error: {e}')
      traceback.print_exc()
      os.kill(os.getpid(), signal.SIGKILL)


def _abort_or_raise(
    context: grpc.ServicerContext | None, code: grpc.StatusCode, details: str
):
  """Safely aborts a gRPC context if available, or raises an Exception."""
  if context is None:
    raise RuntimeError(details)

  context.abort(code, details)


class Driver:
  """Drives the engines."""

  _prefill_engines: list[engine_api.Engine]
  _generate_engines: list[engine_api.Engine]
  # Allows us to pre-load the params, primarily so that we can iterate quickly
  # on the driver in colab without reloading weights.
  _prefill_params: Optional[dict[int, Any]] = {}
  _generate_params: Optional[dict[int, Any]] = {}
  # Stage 1
  _prefill_backlog: queue.Queue[ActiveRequest | None]
  # Stage 2
  # We keep this as a dict to avoid a possibly expensive object comparison
  # when logging the index of the generate engine we send a prefill result
  # to, it allows us to natively have the index from the min operation, rather
  # than have to call .index()
  _generate_backlogs: dict[int, queue.Queue[ActiveRequest | None]] = {}
  # Stage 3
  # This can be a list because we can pass it as an arg to generate and
  # detokenize threads. It is a list of tokens to be detokenized.
  _detokenize_backlogs: list[queue.Queue[engine_api.ResultTokens]] = []
  _generate_slots: list[queue.Queue[int]] = []
  _active_requests: list[queue.Queue[tuple[int, ActiveRequest | None]]] = []

  def __init__(
      self,
      prefill_engines=None,
      generate_engines=None,
      prefill_params=None,
      generate_params=None,
  ):
    logging.info(
        'Initialising driver with %d prefill engines and %d generate engines.',
        len(prefill_engines),
        len(generate_engines),
    )
    self._prefill_engines = prefill_engines
    self._generate_engines = generate_engines
    self._prefill_params = prefill_params if prefill_params else {}
    self._generate_params = generate_params if generate_params else {}
    # Stages 1-4 represent the life cycle of a request.
    # Stage 1
    # At first, a request is placed here in order to get prefilled.
    self._prefill_backlog = queue.Queue()
    # Stage 2
    # Each generate engine accesses its own generate backlog.
    self._generate_backlogs = {
        idx: queue.Queue() for idx, _ in enumerate(generate_engines)
    }
    # Stage 3
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
    # does a detokenisation for any slots which have previously been set active
    # via the previous kind of object, and the int is used to log which step
    # the tokens were created at. By having them in one queue we prevent
    # the possibility of race conditions where a slot is made live before the
    # tokens are ready and it receives tokens from a different sequence,
    # or tokens detokenized before the relevant slot is live.
    self._detokenize_backlogs = [queue.Queue() for _ in generate_engines]
    # Finally we have a queue that doesn't represent a stage, but tracks the
    # internal status of a request within the generation loop.
    # a) A queue of integers representing available 'slots' in the decode
    #   operation. I.e. potentially available rows in the batch and/or
    #   microbatch. When we want to insert a prefill result, we pop an integer
    #   to insert at. When this is empty, it means all slots are full.

    # Construct a)
    self._generate_slots = [queue.Queue() for _ in generate_engines]
    _ = [
        [
            self._generate_slots[idx].put(i)
            for i in range(engine.max_concurrent_decodes)
        ]
        for idx, engine in enumerate(generate_engines)
    ]

    # Kick off all our threads
    self._prefill_threads = [
        JetThread(
            target=functools.partial(self._prefill_thread, idx, engine),
            name=f'prefill-{idx}',
        )
        for idx, engine in enumerate(self._prefill_engines)
    ]
    self._generate_threads = [
        JetThread(
            target=functools.partial(
                self._generate_thread,
                idx,
                engine,
                self._generate_slots[idx],
                self._detokenize_backlogs[idx],
            ),
            name=f'generate-{idx}',
        )
        for idx, engine in enumerate(self._generate_engines)
    ]
    # Construct b)
    self.detokenize_threads = [
        JetThread(
            target=functools.partial(
                self._detokenize_thread,
                idx,
                engine,
                self._generate_slots[idx],
                self._detokenize_backlogs[idx],
            ),
            name=f'detokenize-{idx}',
        )
        for idx, engine in enumerate(self._generate_engines)
    ]
    self._all_threads = list(
        itertools.chain(
            self._prefill_threads,
            self._generate_threads,
            self.detokenize_threads,
        )
    )
    self.live = True
    # Kick off all threads
    for t in self._all_threads:
      t.start()

  def stop(self):
    """Stops the driver and all background threads."""
    # Signal to all threads that they should stop.
    self.live = False

    all_backlogs = list(
        itertools.chain(
            [self._prefill_backlog],
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

  def get_total_concurrent_requests(self) -> int:
    """Returns the total number of concurrent requests the driver can service."""
    # We don't support filling all backlogs at once because it can cause GIL
    # contention.
    total_max_concurrent_decodes = sum(
        [e.max_concurrent_decodes for e in self._generate_engines]
    )
    return total_max_concurrent_decodes

  def place_request_on_prefill_queue(self, request: ActiveRequest):
    """Used to place new requests for prefilling and generation."""
    self._prefill_backlog.put(request)

  def _load_cache_history(self, path: str) -> Union[None, Any]:
    """Loads previous kv cache for a longer conversation."""
    if path:
      raise NotImplementedError
    else:
      return None

  def _prefill_thread(
      self,
      idx: int,
      prefill_engine: engine_api.Engine,
      generate_backpressure: int = 3,
  ):
    """Thread which runs in the background performing prefills."""
    logging.info('---------Spinning up prefill thread %d.---------', idx)
    prefill_params = self._prefill_params[idx]
    metadata = prefill_engine.get_tokenizer()
    vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
    logging.info('---------Prefill params %d loaded.---------', idx)

    while self.live:
      # We don't want to keep lots of kv caches live in memory on the prefill
      # slice that aren't about to be sent over to a generation slice.
      if self._generate_backlogs[idx].qsize() < generate_backpressure:
        # Check if there is anything on the prefill backlog, pop if so.
        try:
          request = self._prefill_backlog.get(block=True)
          if request is None:
            break
          # TODO: Implement hot/cold cache for history.
          history = self._load_cache_history(request.history_path)  # pylint: disable = assignment-from-none
          # Tokenize, and introduce a leading dimension
          is_bos = not bool(request.history_path)
          logging.info(
              'Prefilling on prefill engine %d : prefill queue size, %d,'
              ' is_bos: %s, history: %s',
              idx,
              self._prefill_backlog.qsize(),
              is_bos,
              request.history_path,
          )  # pylint: disable = line-too-long
          padded_tokens, true_length = token_utils.tokenize_and_pad(
              request.prefill_text,
              vocab,
              is_bos=is_bos,
              max_prefill_length=prefill_engine.max_prefill_length,
          )
          # Compute new kv cache for the prefill_text, conditional on
          # history.
          prefill_result = prefill_engine.prefill(
              params=prefill_params,
              existing_prefix=history,
              padded_tokens=padded_tokens,
              true_length=true_length,
          )
          request.prefill_result = prefill_result
          # Once prefill is complete, place it on the generation queue.
          self._generate_backlogs[idx].put(request)
          logging.info(
              'Placed request on the generate queue,'
              f' {self._generate_backlogs[idx].qsize()=}'
          )
        except queue.Empty:
          # Otherwise, don't do anything!
          pass

  def _generate_thread(
      self,
      idx: int,
      generate_engine: engine_api.Engine,
      my_slots: queue.Queue[int],
      my_detokenize_backlog: queue.Queue[tuple[int, engine_api.ResultTokens]],
      backpressure: int = 3,
  ):
    """Step token generation and insert prefills from backlog.

    Args:
      idx: Idx corresponding to which generation engine idx this is.
      generate_engine: The generate engine corresponding to this thread.
      my_slots: Integers representing free my_slots.
      my_detokenize_backlog: Detokenize backlog for this generate engine.
      backpressure: How many steps we can queue up before pausing, allows us to
        hide dispatch times because jit calls are dispatched one after another.
        We have this in the first place because we don't want to enqueue
        thousands of steps.
    """
    logging.info('---------Spinning up generate thread %d.---------', idx)
    # Keep track of what step tokens were generated at.
    generate_timestep = 0
    # State to store things like running kv cache in.
    decode_state = generate_engine.init_decode_state()
    generate_params = self._generate_params[idx]
    logging.info('---------Generate params %d loaded.---------', idx)
    time_of_last_generate = time.time()
    time_of_last_print = time.time()
    while self.live:
      if (time.time() - time_of_last_print) > 1:
        logging.info(
            'Generate thread making a decision with:'
            f' prefill_backlog={self._prefill_backlog.qsize()} generate_free_slots={my_slots.qsize()}'
        )
        time_of_last_print = time.time()
      # Check if there are any free my_slots.
      if not my_slots.empty() and not self._generate_backlogs[idx].empty():
        # Only get requests from the backlog corresponding to this engine.
        new_request = self._generate_backlogs[idx].get()
        if new_request is None:
          break
        slot = my_slots.get()
        logging.info(
            'Generate slice %d slot %d step %d',
            idx,
            slot,
            generate_timestep,
        )
        decode_state = generate_engine.insert(
            new_request.prefill_result, decode_state, slot=slot
        )
        delete_pytree(new_request.prefill_result)
        new_request.generate_timestep_added = generate_timestep
        new_request.complete = np.zeros(
            (generate_engine.samples_per_slot,), dtype=np.bool_
        )
        my_detokenize_backlog.put((slot, new_request))

      if my_detokenize_backlog.qsize() < backpressure:
        decode_state, sampled_tokens = generate_engine.generate(
            generate_params, decode_state
        )
        sampled_tokens.copy_to_host_async()
        my_detokenize_backlog.put((generate_timestep, sampled_tokens))
        generate_timestep += 1
        logging.info(
            'Generate engine %d step %d - slots free : %d / %d, took %.2fms',
            idx,
            generate_timestep,
            my_slots.qsize(),
            generate_engine.max_concurrent_decodes,
            (time.time() - time_of_last_generate) * 10**3,
        )
        time_of_last_generate = time.time()

  def _detokenize_thread(
      self,
      idx: int,
      generate_engine: engine_api.Engine,
      my_slots: queue.Queue[int],
      my_detokenize_backlog: queue.Queue[
          Union[tuple[int, engine_api.ResultTokens], tuple[int, ActiveRequest]]
      ],
  ):
    """Detokenize sampled tokens and returns them to the user."""
    del idx
    # One of these per generate engine.
    # For all filled my_slots, pop the sampled token onto the relevant
    # requests return channel. If it done, place it back onto free slots.
    metadata = generate_engine.get_tokenizer()
    vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

    my_live_requests = {
        i: None for i in range(generate_engine.max_concurrent_decodes)
    }

    while self.live:
      try:
        data = my_detokenize_backlog.get(block=True)
        if data is None:
          break
        start_detokenise_time = time.time()
        if isinstance(data[1], engine_api.ResultTokens):
          # We want to detokenise them.
          generate_timestep_added, result_tokens = data
          # Disable attribute error because pytype doesn't know this
          # is a result tokens, and we can't annotate the tuple.
          result_tokens = result_tokens.convert_to_numpy()  # pytype: disable=attribute-error

          for slot, request in my_live_requests.items():
            if request is not None:
              results, complete = token_utils.process_result_tokens(
                  slot=slot,
                  slot_max_length=request.max_tokens,
                  result_tokens=result_tokens,
                  vocab=vocab,
                  complete=request.complete,
              )
              request.complete = complete
              # Return some tokens.
              request.enqueue_tokens(results)
              if request.complete.all():
                # Place the slot back on the free queue.
                my_live_requests[slot] = None
                my_slots.put(slot)
          logging.info(
              'Detokenising generate step %d took %.2fms',
              generate_timestep_added,
              (time.time() - start_detokenise_time) * 10**3,
          )
        else:
          # We want to update a slot with the new channel.
          slot, active_request = data
          my_live_requests[slot] = active_request

      except queue.Empty:
        # Nothing to detokenize!
        pass


class LLMOrchestrator(jetstream_pb2_grpc.OrchestratorServicer):
  """Coordinates a set of prefill and generate slices for LLM decoding."""

  _driver: Driver

  def __init__(self, driver: Driver):
    self._driver = driver

  def Decode(
      self,
      request: jetstream_pb2.DecodeRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> Iterable[jetstream_pb2.DecodeResponse]:
    """Decode."""
    if context is None:
      logging.warning(
          'LLM orchestrator is being used in offline test mode, and will not'
          ' respond to gRPC queries - only direct function calls.'
      )
    # Wrap request as an ActiveRequest.
    active_request = ActiveRequest(
        max_tokens=request.max_tokens,
        history_path=request.session_cache,
        prefill_text=request.additional_text,
    )
    # The first stage is being prefilled, all other stages are handled
    # inside the driver (transfer, generate*N, detokenize).
    try:
      self._driver.place_request_on_prefill_queue(active_request)
    except queue.Full:
      # Safely abort the gRPC server thread with a retriable error.
      _abort_or_raise(
          context=context,
          code=grpc.StatusCode.RESOURCE_EXHAUSTED,
          details=(
              'The driver prefill queue is full and more requests cannot be'
              ' handled. You may retry this request.'
          ),
      )
    logging.info(
        'Placed request on the prefill queue.',
    )

    while not (active_request.complete and active_request.return_channel.empty()):
      # When an active request is created a queue is instantiated. New tokens
      # are placed there during the decoding loop, we pop from that queue by
      # using the .next method on the active request.
      # Yielding allows for the response to be a streaming grpc call - which
      # can be called via iterating over a for loop on the other side.
      yield jetstream_pb2.DecodeResponse(response=active_request.next())
