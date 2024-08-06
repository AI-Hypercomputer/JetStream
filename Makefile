PYTHON := python
PIP := $(PYTHON) -m pip
GRPC_TOOLS_VERSION := 1.62.1

.PHONY: all update-deps generate-protos check

all: update-deps generate-protos format

update-deps:
	$(PIP) install pip-tools
	$(PYTHON) -m piptools compile requirements.in

generate-protos: generate-and-append-preambles format

format:
	$(PIP) install pyink
	pyink --pyink-indentation 2 --line-length 80 --verbose .

check:
	pylint --ignore-patterns=".*_pb2.py,.*_pb2_grpc.py" jetstream/ benchmarks/

generate-and-append-preambles:
	$(PIP) install grpcio-tools==$(GRPC_TOOLS_VERSION)
	for id in $$(find . -name "*.proto"); do \
		$(PYTHON) -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. $$id && \
		PROTO_FILE=$$(echo $$id | awk '{print substr($$0, 1, length($$0)-6)}') && \
		PB_GRPC_PY=$(addsuffix "_pb2_grpc.py",$$PROTO_FILE) && \
		PB_PY=$(addsuffix "_pb2.py",$$PROTO_FILE) && \
		cat license_preamble.txt $$PB_GRPC_PY >> $(addsuffix "_temp",$$PB_GRPC_PY) && \
		mv $(addsuffix "_temp",$$PB_GRPC_PY) $$PB_GRPC_PY; \
		cat license_preamble.txt $$PB_PY >> $(addsuffix "_temp",$$PB_PY) && \
		mv $(addsuffix "_temp",$$PB_PY) $$PB_PY; \
	done

