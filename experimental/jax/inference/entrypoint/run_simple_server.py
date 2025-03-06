"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import uvicorn
import os


if __name__ == "__main__":
  print("start")
  current_dir = os.path.dirname(__file__)
  parent_dir = os.path.dirname(current_dir)

  uvicorn.run(
      app_dir=f"{parent_dir}/server",
      app="simple_server:app",
      host="0.0.0.0",
      port=8000,
  )
