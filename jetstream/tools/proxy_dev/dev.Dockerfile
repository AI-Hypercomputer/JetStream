# Ubuntu:22.04
# Use Ubuntu 22.04 from Docker Hub.
# https://hub.docker.com/_/ubuntu/tags\?page\=1\&name\=22.04
FROM base_image

ENV DEBIAN_FRONTEND=noninteractive

ENV JAX_PLATFORMS=proxy
ENV JAX_BACKEND_TARGET=grpc://localhost:38681

# Copy all files from local workspace into docker container
COPY  JetStream ./JetStream
COPY  maxtext ./maxtext

RUN pip install ./JetStream

ENTRYPOINT ["bash"]
