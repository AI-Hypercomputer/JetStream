# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""mlperf loadgen interface for LLama2."""
import array
import concurrent.futures
import dataclasses
import json
import logging
from operator import itemgetter  # pylint: disable=g-importing-member
import time
from typing import List, Optional, Any

import numpy as np

import dataset

import mlperf_loadgen as lg

import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc

from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend.py")

_last_log_time = 0  # Global variable to track last log time


def _log_once_n_sec(msg: str, log_interval_sec: int):
  """Logs a message once every specified number of seconds."""
  global _last_log_time
  current_time = time.time()
  if current_time - _last_log_time >= log_interval_sec:
    log.info("(once every %dsec): %s", log_interval_sec, msg)
    _last_log_time = current_time


@dataclasses.dataclass
class WarmupSample:
  id: int
  index: int


@dataclasses.dataclass
class StreamResponse:
  result: str = ""


class ThreadedLMClient:
  """Holds a thread pool and a loadgen client for LM inference."""

  _thread_pool: concurrent.futures.ThreadPoolExecutor
  _dataset: dataset.Dataset
  _futures = List[concurrent.futures.Future]

  def __init__(
      self,
      is_stream: bool,
      num_threads: int,
      api_url: str,
      dataset_object: dataset.Dataset,
      input_mode: str,
      output_mode: str,
      tokenizer: Optional[AutoTokenizer] = None,
      max_output_len: int = 1024,
      log_interval: int = 1000,
  ):
    log.info(f"Initiating {self.__class__.__name__} ...")
    self._is_stream = is_stream
    self._input_mode = dataset.validate_sample_mode(input_mode)
    self._output_mode = dataset.validate_sample_mode(output_mode)
    if self._input_mode == "text" or self._output_mode == "text":
      assert tokenizer is not None
    self._tokenizer = tokenizer
    self._max_output_len = max_output_len

    self._log_interval = log_interval

    self._thread_pool = concurrent.futures.ThreadPoolExecutor(num_threads)
    self._api_url = api_url
    self._dataset = dataset_object
    self._futures = []
    self.pred_outputs = {}
    self._resp_cnt = 0

    log.info("Creating grpc channel with api_url {}".format(api_url))
    options = [("grpc.keepalive_timeout_ms", 10000)]
    self._grpc_channel = grpc.insecure_channel(api_url, options=options)

  @property
  def tokenizer(self):
    return self._tokenizer

  def _log_resp_cnt(self):
    self._resp_cnt += 1
    if self._resp_cnt % self._log_interval == 0:
      log.info("Completed %d queries", self._resp_cnt)

  def process_single_sample_async(self, query_sample, warmup):
    """Executes a single query and marks responses complete asynchronously.

    Args:
      query_sample: Single prompt
      warmup: Indicates that this is a warmup request.
    """
    future = self._thread_pool.submit(
        self._process_sample, query_sample, warmup
    )
    self._futures.append(future)

  def flush(self):
    concurrent.futures.wait(self._futures)
    self._futures = []

  def _grpc_request(self, request, sample, warmup):
    """Send grpc synchronous request since the current grpc server is sync."""
    stub = jetstream_pb2_grpc.OrchestratorStub(self._grpc_channel)
    token_list = []
    ttft = 0
    start_time = time.perf_counter()
    response = stub.Decode(request)
    for resp in response:
      if not warmup and self._is_stream and ttft == 0:
        # TTFT for online mode
        ttft = time.perf_counter() - start_time
        _log_once_n_sec("TTFRT {}ms".format(ttft * 1000), 30)
        response_token_ids = resp.stream_content.samples[0].token_ids
        assert len(response_token_ids) == 1
        response_token_ids = np.array(response_token_ids, dtype=np.int64)
        response_array = array.array("B", response_token_ids.tobytes())
        response_info = response_array.buffer_info()
        first_token_response = lg.QuerySampleResponse(
            sample.id, response_info[0], response_info[1]
        )
        lg.FirstTokenComplete([first_token_response])
      token_list.extend(resp.stream_content.samples[0].token_ids)
    return token_list

  def _process_sample(self, sample, warmup):
    """Processes a single sample."""
    sample_data = self._dataset.inputs[sample.index]
    if self._input_mode == "text":
      token_ids = self._tokenizer.encode(sample_data)
    else:
      assert self._input_mode == "tokenized"
      token_ids = [int(token_id_str) for token_id_str in sample_data.split(",")]

    request = jetstream_pb2.DecodeRequest(
        token_content=jetstream_pb2.DecodeRequest.TokenContent(
            token_ids=token_ids
        ),
        max_tokens=self._max_output_len,
    )
    generated_token_list = self._grpc_request(request, sample, warmup)
    if not warmup:
      response_token_ids = generated_token_list
      n_tokens = len(response_token_ids)
      response_token_ids = np.array(response_token_ids, dtype=np.int64)
      response_array = array.array("B", response_token_ids.tobytes())
      response_info = response_array.buffer_info()
      response_data = response_info[0]
      response_size = response_info[1] * response_array.itemsize
      query_sample_response = lg.QuerySampleResponse(
          sample.id, response_data, response_size, n_tokens
      )
      lg.QuerySamplesComplete([query_sample_response])
      _log_once_n_sec("Mark query complete", 30)

      pred_output = self._tokenizer.decode(response_token_ids)
      self.pred_outputs[sample.index] = pred_output
      self._log_resp_cnt()


class SUT:
  """SUT."""

  def __init__(
      self,
      scenario,
      api_url,
      is_stream,
      input_mode,
      output_mode,
      max_output_len,
      dataset_path,
      total_sample_count,
      tokenizer_path=None,
      perf_count_override=None,
      num_client_threads=200,
      log_interval=1000,
      batch_size_exp=5,
      pred_outputs_log_path=None,
      dataset_rename_cols="",
  ):
    log.info(f"Starting {scenario} SUT with {api_url}.")
    self._is_stream = is_stream
    self._input_mode = dataset.validate_sample_mode(input_mode)
    self._output_mode = dataset.validate_sample_mode(output_mode)
    assert tokenizer_path is not None
    self._tokenizer = self.load_tokenizer(tokenizer_path)
    self._max_output_len = max_output_len
    self._api_url = api_url
    self._dataset_path = dataset_path
    self._total_sample_count = total_sample_count
    self._perf_count_override = perf_count_override
    self._num_client_threads = num_client_threads
    self._log_interval = log_interval
    self._batch_size_exp = batch_size_exp
    self._pred_outputs_log_path = pred_outputs_log_path

    log.info("Loading Dataset ... ")
    self.dataset = dataset.Dataset(
        dataset_path=self._dataset_path,
        input_mode=self._input_mode,
        total_sample_count=self._total_sample_count,
        perf_count_override=self._perf_count_override,
        dataset_rename_cols=dataset_rename_cols,
    )

    client_cls = ThreadedLMClient
    self._client = client_cls(
        is_stream=self._is_stream,
        num_threads=self._num_client_threads,
        api_url=self._api_url,
        dataset_object=self.dataset,
        input_mode=self._input_mode,
        output_mode=self._output_mode,
        tokenizer=self._tokenizer,
        max_output_len=self._max_output_len,
        log_interval=self._log_interval,
    )

    self.qsl = lg.ConstructQSL(
        self.dataset.total_sample_count,
        self.dataset.perf_count,
        self.dataset.LoadSamplesToRam,
        self.dataset.UnloadSamplesFromRam,
    )

    # We need to add some warmup to improve throughput estimation
    log.info("Starting warmup....")
    # Warm up with exponentially increasing batch sizes up to 32.
    for batch_size_exp in range(self._batch_size_exp):
      batch_size = 2**batch_size_exp
      for warmup_id, warmup_idx in enumerate(range(batch_size)):
        warmup_sample = WarmupSample(id=warmup_id, index=warmup_idx)
        self._client.process_single_sample_async(warmup_sample, True)
      self._client.flush()

    log.info("Warmup done....")
    time.sleep(30)
    self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

  def load_tokenizer(
      self, tokenizer_path: Optional[str] = None
  ) -> Optional[AutoTokenizer]:
    """Returns tokenizer"""
    if tokenizer_path is not None:
      tokenizer = AutoTokenizer.from_pretrained(
          tokenizer_path,
          model_max_length=1024,
          padding_side="left",
          use_fast=True,
      )
      tokenizer.pad_token = tokenizer.eos_token
      return tokenizer

  def _sort_issue_queries(self, query_samples):
    """Issue queries."""
    query_samples_with_length = []
    for query_sample in query_samples:
      query_sample_token_length = self.dataset.inputs_with_token_lengths[
          query_sample.index
      ][1]
      query_samples_with_length.append(
          (query_sample_token_length, query_sample)
      )
    sorted_query_samples_with_length = sorted(
        query_samples_with_length, key=itemgetter(0)
    )
    sorted_query_samples = [x[1] for x in sorted_query_samples_with_length]
    return sorted_query_samples

  def issue_queries(self, query_samples):
    """Issue queries."""
    num_query_samples = len(query_samples)
    if num_query_samples > 1:
      log.info(f"Issuing {num_query_samples} queries. ")
      query_samples = self._sort_issue_queries(query_samples)
    for query_sample in query_samples:
      self._client.process_single_sample_async(query_sample, False)

  def flush_queries(self):
    """Flush queries."""
    log.info("Loadgen has completed issuing queries... ")
    self._client.flush()

    if self._pred_outputs_log_path is not None:

      pred_outputs = []
      for idx, x in self._client.pred_outputs.items():
        pred_output = {
            "qsl_idx": idx,
            "intput": self._client._dataset.inputs[idx],
            "data": x,
        }
        pred_outputs.append(pred_output)
      log.info(f"Generated {len(pred_outputs)} prediction outputs")

      if pred_outputs:
        self.accuracy_log = open(self._pred_outputs_log_path, "w")
        self.accuracy_log.write(json.dumps(pred_outputs))
        self.accuracy_log.flush()
        self.accuracy_log.close()
        log.info("Dumpped prediction outputs to accuracy log... ")

  def __del__(self):
    print("Finished destroying SUT.")
