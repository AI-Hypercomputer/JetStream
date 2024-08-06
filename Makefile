PYTHON := python
PIP := $(PYTHON) -m pip
GRPC_TOOLS_VERSION := 1.62.1

.PHONY: all update-deps generate-protos check

all: update-deps generate-protos format

update-deps:
	$(PIP) install pip-tools
	$(PYTHON) -m piptools compile requirements.in

generate-protos: generate format

generate:
	$(PIP) install grpcio-tools==$(GRPC_TOOLS_VERSION)
	$(PYTHON) -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. jetstream/core/proto/jetstream.proto

	cat license_preamble.txt jetstream/core/proto/jetstream_pb2_grpc.py >> jetstream/core/proto/jetstream_pb2_grpc.py_temp && \
	mv jetstream/core/proto/jetstream_pb2_grpc.py_temp jetstream/core/proto/jetstream_pb2_grpc.py

	cat license_preamble.txt jetstream/core/proto/jetstream_pb2.py >> jetstream/core/proto/jetstream_pb2.py_temp && \
	mv jetstream/core/proto/jetstream_pb2.py_temp jetstream/core/proto/jetstream_pb2.py

format:
	$(PIP) install pyink
	pyink --pyink-indentation 2 --line-length 80 --verbose .

check:
	pylint --ignore-patterns=".*_pb2.py,.*_pb2_grpc.py" jetstream/ benchmarks/

