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

COPY inference_mlperf4.1 ./inference_mlperf4.1
RUN apt-get -y install python3-dev && apt-get -y install build-essential
RUN pip install ./inference_mlperf4.1/loadgen
RUN pip install \
    transformers==4.31.0 \
    nltk==3.8.1 \
    evaluate==0.4.0 \
    absl-py==1.4.0 \
    rouge-score==0.1.2 \
    sentencepiece==0.1.99 \
    accelerate==0.21.0

ENTRYPOINT ["bash"]
