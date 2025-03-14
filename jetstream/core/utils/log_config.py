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

import logging
import sys
import os


def get_logger(name: str, log_level: int = logging.WARN) -> logging.Logger:
  """Configures and returns a structured logger.

  Args:
      name: The name of the logger.
      log_level: The logging level (e.g., logging.DEBUG, logging.INFO).

  Returns:
      logging.Logger: Configured logger instance.
  """

  logger = logging.getLogger(name)
  logger.setLevel(log_level)
  logger.propagate = False
  if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

  return logger
