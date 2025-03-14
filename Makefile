PYTHON := python
PIP := $(PYTHON) -m pip
GRPC_TOOLS_VERSION := 1.62.1

all: install-deps generate-protos format check

# Dependency management targets
install-deps:
	$(PIP) install pytype pylint pyink -r requirements.txt -r benchmarks/requirements.in

# Code generation/formatting targets
generate-protos: generate-and-prepend-preambles format

generate-and-prepend-preambles:
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

format:
	$(PIP) install pyink
	pyink --pyink-indentation 2 --line-length 80 --verbose .

# Code checking related targets
check: type-check format-check linter-check

type-check:
	$(PIP) install pytype
	pytype --jobs auto --disable=import-error,module-attr jetstream/ benchmarks/ --exclude='benchmarks/.*\.json$\'

format-check:
	$(PIP) install pyink
	pyink --pyink-indentation 2 --line-length 80 --check --verbose .

linter-check:
	$(PIP) install pylint
	pylint --ignore-patterns=".*_pb2.py,.*_pb2_grpc.py" jetstream/ benchmarks/


# Testing related targets
tests: unit-tests check-test-coverage

unit-tests:
	coverage run -m unittest -v

check-test-coverage:
	coverage report -m --omit="jetstream/core/proto/*,jetstream/engine/tokenizer_pb2.py,jetstream/external_tokenizers/*,benchmarks/benchmark_serving.py,benchmarks/eval_accuracy.py,benchmarks/eval_accuracy_mmlu.py,benchmarks/math_utils.py" --fail-under=96
