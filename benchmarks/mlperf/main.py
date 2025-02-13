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

import argparse

import gc
import logging
import os
import sys

import backend

import mlperf_loadgen as lg

_MLPERF_ID = "llama2-70b"

sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main.py")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--scenario",
      type=str,
      choices=["Offline", "Server"],
      default="Offline",
      help="Scenario",
  )
  parser.add_argument(
      "--api-url", type=str, default=None, help="SAX published model path."
  )
  parser.add_argument("--dataset-path", type=str, default=None, help="")
  parser.add_argument("--tokenizer-path", type=str, default=None, help="")
  parser.add_argument(
      "--accuracy", action="store_true", help="Run accuracy mode"
  )
  parser.add_argument("--is-stream", action="store_true", help="")
  parser.add_argument(
      "--input-mode",
      type=str,
      choices=["text", "tokenized"],
      default="tokenized",
  )
  parser.add_argument(
      "--output-mode",
      type=str,
      choices=["text", "tokenized"],
      default="tokenized",
  )
  parser.add_argument(
      "--max-output-len", type=int, default=1024, help="Maximum output len"
  )
  parser.add_argument(
      "--audit-conf",
      type=str,
      default="audit.conf",
      help="audit config for LoadGen settings during compliance runs",
  )
  parser.add_argument(
      "--mlperf-conf",
      type=str,
      default="mlperf.conf",
      help="mlperf rules config",
  )
  parser.add_argument(
      "--user-conf",
      type=str,
      default="user.conf",
      help="user config for user LoadGen settings such as target QPS",
  )
  parser.add_argument(
      "--total-sample-count",
      type=int,
      default=24576,
      help="Number of samples to use in benchmark.",
  )
  parser.add_argument(
      "--perf-count-override",
      type=int,
      default=None,
      help="Overwrite number of samples to use in benchmark.",
  )
  parser.add_argument(
      "--output-log-dir",
      type=str,
      default="output-logs",
      help="Where logs are saved.",
  )
  parser.add_argument(
      "--enable-log-trace",
      action="store_true",
      help="Enable log tracing. This file can become quite large",
  )
  parser.add_argument(
      "--num-client-threads",
      type=int,
      default=200,
      help="Number of client threads to use",
  )
  parser.add_argument("--batch-size-exp", type=int, default=6, help="")
  parser.add_argument("--log-pred-outputs", action="store_true", help="")
  parser.add_argument(
      "--log-interval",
      type=int,
      default=1000,
      help="Logging interval in seconds",
  )
  parser.add_argument(
      "--user-conf-override-path",
      type=str,
      default="",
      help="When given overrides the default user.conf path",
  )
  parser.add_argument(
      "--rename-dataset-cols",
      type=str,
      default="",
      help=(
          "Rename some of the dataset columns to whats expected by code. For example, "
          "mixtral dataset uses ref_token_length instead of ref_token_len. Format is a string dict "
          'eg. {"tok_input_len": "tok_input_length"}'
      ),
  )
  parser.add_argument(
      "--mlperf-conf-id",
      type=str,
      default=_MLPERF_ID,
      help="When given overrides the default user.conf path",
  )
  args = parser.parse_args()
  return args


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}


def main():
  args = get_args()

  settings = lg.TestSettings()
  settings.scenario = scenario_map[args.scenario.lower()]
  if args.user_conf_override_path:
    user_conf = args.user_conf_override_path
  else:
    user_conf = args.user_conf

  settings.FromConfig(user_conf, args.mlperf_conf_id, args.scenario)
  log.info("Mlperf config: %s", args.mlperf_conf)
  log.info("User config: %s", user_conf)

  if args.accuracy:
    settings.mode = lg.TestMode.AccuracyOnly
    log.warning(
        "Accuracy run will generate the accuracy logs, but the evaluation of the log is not completed yet"
    )
  else:
    settings.mode = lg.TestMode.PerformanceOnly
    settings.print_timestamps = True

  settings.use_token_latencies = True

  os.makedirs(args.output_log_dir, exist_ok=True)
  log_output_settings = lg.LogOutputSettings()
  log_output_settings.outdir = args.output_log_dir
  log_output_settings.copy_summary_to_stdout = True
  log_settings = lg.LogSettings()
  log_settings.log_output = log_output_settings
  log_settings.enable_trace = args.enable_log_trace

  sut = backend.SUT(
      scenario=args.scenario.lower(),
      api_url=args.api_url,
      is_stream=args.is_stream,
      input_mode=args.input_mode,
      output_mode=args.output_mode,
      max_output_len=args.max_output_len,
      dataset_path=args.dataset_path,
      total_sample_count=args.total_sample_count,
      tokenizer_path=args.tokenizer_path,
      perf_count_override=args.perf_count_override,
      num_client_threads=args.num_client_threads,
      log_interval=args.log_interval,
      batch_size_exp=args.batch_size_exp,
      pred_outputs_log_path=os.path.join(
          args.output_log_dir, "pred_outputs_logger.json"
      )
      if args.log_pred_outputs
      else None,
      dataset_rename_cols=args.rename_dataset_cols,
  )

  lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
  log.info("Starting Benchmark run")
  lg.StartTestWithLogSettings(
      lgSUT, sut.qsl, settings, log_settings, args.audit_conf
  )

  log.info("Run Completed!")

  log.info("Destroying SUT...")
  lg.DestroySUT(lgSUT)

  log.info("Destroying QSL...")
  lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
  # Disable garbage collection to avoid stalls when running tests.
  gc.disable()
  main()
