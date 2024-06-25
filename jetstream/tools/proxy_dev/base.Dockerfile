# Ubuntu:22.04
# Use Ubuntu 22.04 from Docker Hub.
# https://hub.docker.com/_/ubuntu/tags\?page\=1\&name\=22.04
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y update && apt install -y --no-install-recommends apt-transport-https ca-certificates gnupg git python3.10 python3-pip curl

RUN update-alternatives --install     /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-sdk -y


# Copy all files from local workspace into docker container
COPY  JetStream ./JetStream
COPY  maxtext ./maxtext

RUN cd maxtext/ && \
pip install -r requirements.txt

RUN pip install setuptools==58 fastapi==0.103.2 uvicorn nltk evaluate

RUN pip install ./JetStream

ENTRYPOINT ["bash"]
