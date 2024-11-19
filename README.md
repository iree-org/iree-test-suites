# iree-test-suites

Test suites for IREE and related projects.

## Test suites

### [linalg_ops/](linalg_ops/) : 'linalg' and related operations

[![Test Linalg Ops](https://github.com/iree-org/iree-test-suites/actions/workflows/test_linalg_ops.yml/badge.svg?branch=main)](https://github.com/iree-org/iree-test-suites/actions/workflows/test_linalg_ops.yml?query=branch%3Amain)

* Generated tests for matrix multiplication, convolution, and attention using
  the [MLIR 'linalg' dialect](https://mlir.llvm.org/docs/Dialects/Linalg/) and
  the
  [IREE 'linalg_ext' dialect](https://iree.dev/reference/mlir-dialects/LinalgExt/).
* Built with [cmake](https://cmake.org/) and run via
  [ctest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) (for now?).

### [onnx_models/](onnx_models/) : Open Neural Network Exchange models

[![Test ONNX Models](https://github.com/iree-org/iree-test-suites/actions/workflows/test_onnx_models.yml/badge.svg?branch=main)](https://github.com/iree-org/iree-test-suites/actions/workflows/test_onnx_models.yml?query=branch%3Amain)

* Tests that import, compile, and run ONNX models through IREE then compare
  the outputs against a reference (ONNX Runtime).
* Runnable via [pytest](https://docs.pytest.org/).

### [onnx_ops/](onnx_ops/) : Open Neural Network Exchange operations

[![Test ONNX Ops](https://github.com/iree-org/iree-test-suites/actions/workflows/test_onnx_ops.yml/badge.svg?branch=main)](https://github.com/iree-org/iree-test-suites/actions/workflows/test_onnx_ops.yml?query=branch%3Amain)

* 1250+ tests for [ONNX](https://onnx.ai/) framework
  [operators](https://onnx.ai/onnx/operators/).
* Runnable via [pytest](https://docs.pytest.org/) using a
  configurable set of flags to `iree-compile` and `iree-run-module`.
