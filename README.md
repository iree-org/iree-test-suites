# iree-test-suites

Test suites for IREE and related projects.

🚧🚧🚧 Under construction 🚧🚧🚧

See https://groups.google.com/g/iree-discuss/c/GIWyj8hmP0k/ for context.

## Test suites

### ONNX Ops ([onnx-ops/](onnx-ops/))

[![Test ONNX Ops](https://github.com/iree-org/iree-test-suites/actions/workflows/test_onnx_ops.yml/badge.svg?branch=main)](https://github.com/iree-org/iree-test-suites/actions/workflows/test_onnx_ops.yml?query=branch%3Amain)

* 1250+ tests for ONNX (Open Neural Network Exchange: https://onnx.ai/)
  operators (https://onnx.ai/onnx/operators/).
* Runnable via [pytest](https://docs.pytest.org/en/stable/) using a
  configurable set of flags to `iree-compile` and `iree-run-module`, allowing
  for testing across different compilation targets and runtime devices.
