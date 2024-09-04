# iree-test-suites

Test suites for IREE and related projects.

🚧🚧🚧 Under construction 🚧🚧🚧

See https://groups.google.com/g/iree-discuss/c/GIWyj8hmP0k/ for context.

## Test suites

### [linalg_ops/](linalg_ops/) : 'linalg' and related ops

[![Test Linalg Ops](https://github.com/iree-org/iree-test-suites/actions/workflows/test_linalg_ops.yml/badge.svg?branch=main)](https://github.com/iree-org/iree-test-suites/actions/workflows/test_linalg_ops.yml?query=branch%3Amain)

* Generated tests for matrix multiplication using the
  [MLIR 'linalg' dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
  (tests for other ops like 'attention' and 'convolution' are planned).
* Built with [cmake](https://cmake.org/) and run via
  [ctest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) (for now?).

### [onnx_ops/](onnx_ops/) : Open Neural Network Exchange operations

[![Test ONNX Ops](https://github.com/iree-org/iree-test-suites/actions/workflows/test_onnx_ops.yml/badge.svg?branch=main)](https://github.com/iree-org/iree-test-suites/actions/workflows/test_onnx_ops.yml?query=branch%3Amain)

* 1250+ tests for [ONNX](https://onnx.ai/) framework
  [operators](https://onnx.ai/onnx/operators/).
* Runnable via [pytest](https://docs.pytest.org/en/stable/) using a
  configurable set of flags to `iree-compile` and `iree-run-module`.
