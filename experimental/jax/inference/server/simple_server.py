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

"""Server module"""

from contextlib import asynccontextmanager
import queue
import uvicorn
from inference import parallel
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from inference.config import Config, ModelId
from inference.runtime.engine import *
from inference.runtime.request_type import *


@asynccontextmanager
async def lifespan(app: FastAPI):
  devices = jax.devices()
  mesh = parallel.create_device_mesh(
      devices=devices,
      shape=(len(devices), 1),
  )
  loop = asyncio.get_running_loop()
  app.state.req_queue = queue.Queue()
  print("starting engine")
  engine = Engine(
      mesh=mesh,
      inference_params=Config.get(ModelId.llama_2_7b_chat_hf),
      mode=EngineMode.ONLINE,
      channel=OnlineChannel(
          req_queue=app.state.req_queue,
          aio_loop=loop,
      ),
      debug_mode=False,
  )
  engine.start()
  yield
  # Clean up the ML models and release the resources
  engine.stop()


app = FastAPI(lifespan=lifespan)


async def streaming_tokens(res_queue):
  while True:
    res: Response | None = await res_queue.get()
    if not res:
      return
    yield res.generated_text


class GenerateRequest(BaseModel):
  prompt: str


@app.post("/generate")
async def generate(req: GenerateRequest):
  res_queue = asyncio.Queue()
  app.state.req_queue.put_nowait(
      OnlineRequest(
          prompt=req.prompt,
          res_queue=res_queue,
      )
  )
  return StreamingResponse(streaming_tokens(res_queue))


if __name__ == "__main__":
  print("start")
  uvicorn.run("simple_server:app", host="0.0.0.0", port=8000)
