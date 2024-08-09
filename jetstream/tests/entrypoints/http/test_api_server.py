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

"""Tests http server end-to-end."""

import subprocess
import sys
import time
import unittest


import requests


class HTTPServerTest(unittest.IsolatedAsyncioTestCase):

  @classmethod
  def setUpClass(cls):
    """Sets up a JetStream http server for unit tests."""
    cls.base_url = "http://localhost:8080"
    cls.server = subprocess.Popen(
        [
            "python",
            "-m",
            "jetstream.entrypoints.http.api_server",
            "--config=InterleavedCPUTestServer",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    time.sleep(10)

  @classmethod
  def tearDownClass(cls):
    """Stop the server gracefully."""
    cls.server.terminate()

  async def test_root_endpoint(self):
    response = requests.get(self.base_url + "/", timeout=5)
    assert response.status_code == 200
    expected_data = {"message": "JetStream HTTP Server"}
    assert response.json() == expected_data

  async def test_health_endpoint(self):
    response = requests.get(self.base_url + "/v1/health", timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert "is_live" in data
    assert data["is_live"] == "True"

  async def test_generate_endpoint(self):
    # Prepare a sample request (replace with actual data)
    sample_request_data = {
        "max_tokens": 10,
        "text_content": {"text": "translate this to french: hello world"},
    }

    response = requests.post(
        self.base_url + "/v1/generate",
        json=sample_request_data,
        stream=True,
        timeout=5,
    )
    assert response.status_code == 200
    full_response = []
    for chunk in response.iter_content(
        chunk_size=None
    ):  # chunk_size=None for complete lines
      if chunk:
        stream_response = chunk.decode("utf-8")
        print(f"{stream_response=}")
        full_response.append(stream_response)
    assert len(full_response) == 11  # 10 tokens + eos token
