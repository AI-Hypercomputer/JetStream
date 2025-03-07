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

"""Manages the list of fine-tuned adapters loaded on top of the base model for serving.
"""

import logging
import grpc
import time
import uuid

from typing import Any, AsyncIterator, Optional, Tuple, cast
from jetstream.core import orchestrator
from jetstream.core.lora import adapter_tensorstore
from jetstream.core.proto import multi_lora_decoding_pb2_grpc
from jetstream.core.proto import multi_lora_decoding_pb2
from jetstream.core.utils import async_multifuture
from jetstream.core.utils.return_sample import ReturnSample
from jetstream.engine import engine_api, tokenizer_api, token_utils


class MultiLoraManager(multi_lora_decoding_pb2_grpc.v1Servicer):
  """Manages the parameters of multiple lora requests and their status/lifetimes."""

  _driver: orchestrator.Driver

  def __init__(self, driver: orchestrator.Driver):
    self._driver = driver

  def models(
      self,
	    request: multi_lora_decoding_pb2.ListAdaptersRequest,
	    context: Optional[grpc.aio.ServicerContext] = None,
  ) -> multi_lora_decoding_pb2.ListAdaptersResponse:
    """ListAdapters all loaded LoRA adapters."""

    try:
      adapters = self._driver.listAdaptersFromTensorstore()

      adapter_infos = []
      for adapter_id, adapter_data in adapters.items():
        if adapter_data.status == "loaded_hbm":
          loading_cost = 0
        elif adapter_data.status == "loaded_cpu":
          loading_cost = 1
        elif adapter_data.status == "unloaded":
          loading_cost = 2
        else:
          loading_cost = -1

        adapter_info = multi_lora_decoding_pb2.AdapterInfo(
            adapter_id=adapter_id,
            loading_cost=loading_cost,
            size_hbm=adapter_data.size_hbm,
            size_cpu=adapter_data.size_cpu,
            last_accessed=adapter_data.last_accessed,
            status=adapter_data.status)

        adapter_infos.append(adapter_info)

      return multi_lora_decoding_pb2.ListAdaptersResponse(success=True, adapter_infos=adapter_infos)
    except Exception as e:
      logging.info(f"Listing of adapters failed with error: {str(e)}")
      return multi_lora_decoding_pb2.ListAdaptersResponse(success=False, error_message=str(e))


  def load_lora_adapter(
      self,
      request: multi_lora_decoding_pb2.LoadAdapterRequest,
      context: Optional[grpc.aio.ServicerContext] = None,
  ) -> multi_lora_decoding_pb2.LoadAdapterResponse:
    """Load a LoRA adapter as mentioned in the request."""

    try:
      self._driver.loadAdapterToTensorstore(request.adapter_id, request.adapter_path)

      return multi_lora_decoding_pb2.LoadAdapterResponse(success=True)
    except Exception as e:
      logging.info(f"Loading of adapter_id={request.adapter_id} failed with error: {str(e)}")
      return multi_lora_decoding_pb2.LoadAdapterResponse(success=False, error_message=str(e))


  def unload_lora_adapter(
      self,
      request: multi_lora_decoding_pb2.UnloadAdapterRequest,
      context: Optional[grpc.aio.ServicerContext] = None,
  ) -> multi_lora_decoding_pb2.UnloadAdapterResponse:
    """Unload a LoRA adapter as mentioned in the request."""
    
    try:
      self._driver.unloadAdapterFromTensorstore(request.adapter_id)
      return multi_lora_decoding_pb2.UnloadAdapterResponse(success=True)
    except Exception as e:
      logging.info(f"Loading of adapter_id={request.adapter_id} failed with error: {str(e)}")
      return multi_lora_decoding_pb2.UnloadAdapterResponse(success=False, error_message=str(e))


  def _get_prefill_content(
      self, request: multi_lora_decoding_pb2.CompletionRequest
  ) -> Tuple[str | list[int], bool]:
    which_content = request.WhichOneof("content")
    content = getattr(request, which_content)
    if which_content == "text_content":
      return cast(multi_lora_decoding_pb2.CompletionRequest.TextContent, content).text, False
    else:
      return (
          list(
              cast(multi_lora_decoding_pb2.CompletionRequest.TokenContent, content).token_ids
          ),
          True,
      )

  def process_client_side_tokenization_response(self, response: Any):
    samples = []
    for sample in response:
      samples.append(
          multi_lora_decoding_pb2.CompletionResponse.StreamContent.Sample(
              token_ids=sample.token_ids,
          )
      )
    return multi_lora_decoding_pb2.CompletionResponse(
        stream_content=multi_lora_decoding_pb2.CompletionResponse.StreamContent(
            samples=samples
        )
    )

  def should_buffer_response(self, response: Any) -> bool:
    for item in response:
      if item.text and token_utils.is_byte_token(item.text[-1]):
        # If any sample ends in bytes, this means we might still need to
        # decode more bytes to compose the string.
        return True

  def process_server_side_tokenization_response(
      self, response: Any, buffered_response_list
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
          multi_lora_decoding_pb2.CompletionResponse.StreamContent.Sample(
              text=token_utils.text_tokens_to_str(text),
              token_ids=token_ids,
          )
      )
    return multi_lora_decoding_pb2.CompletionResponse(
        stream_content=multi_lora_decoding_pb2.CompletionResponse.StreamContent(
            samples=samples
        )
    )

  async def completions(  # pylint: disable=invalid-overridden-method
      self,
      request: multi_lora_decoding_pb2.CompletionRequest,
      context: Optional[grpc.aio.ServicerContext] = None,
  ) -> AsyncIterator[multi_lora_decoding_pb2.CompletionResponse]:

    """Decode."""
    if context is None:
      logging.warning(
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
    active_request = orchestrator.ActiveRequest(
        request_id=uuid.uuid4(),
        max_tokens=request.max_tokens,
        prefill_content=prefill_content,
        is_client_side_tokenization=is_client_side_tokenization,
        return_channel=return_channel,
        adapter_id=request.adapter_id,
        metadata=orchestrator.ActiveRequestMetadata(
            start_time=request.metadata.start_time,
            prefill_enqueue_time=time.perf_counter(),
        ),
    )
    # The first stage is being prefilled, all other stages are handled
    # inside the driver (transfer, generate*N, detokenize).
    try:
      self._driver.place_request_on_prefill_queue(active_request)
    except queue.Full:
      # Safely abort the gRPC server thread with a retriable error.
      await _abort_or_raise(
          context=context,
          code=grpc.StatusCode.RESOURCE_EXHAUSTED,
          details=(
              "The driver prefill queue is full and more requests cannot be"
              " handled. You may retry this request."
          ),
      )
    logging.info(
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
      if is_client_side_tokenization:
        # If is_client_side_tokenization, the client should request with token
        # ids, and the JetStream server will return token ids as response.
        # The client should take care of tokenization and detokenization.
        yield self.process_client_side_tokenization_response(response)
      else:
        # Buffer response mechanism is used to handle streaming
        # detokenization with special character (For some edge cases with
        # SentencePiece tokenizer, it requires to decode a complete sequence
        # instead of a single token).
        if self.should_buffer_response(response):
          buffered_response_list.append(response)
          continue
        yield self.process_server_side_tokenization_response(
            response, buffered_response_list
        )
        # Reset buffer after flushed.
        buffered_response_list = []


