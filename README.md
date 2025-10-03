# iree-test-suites

Test suites for IREE and related projects.

## Test suites

### [litert_models/](litert_models/): LiteRT models

[![Test LiteRT Models](https://github.com/iree-org/iree-test-suites/actions/workflows/test_litert_models.yml/badge.svg?branch=main)](https://github.com/iree-org/iree-test-suites/actions/workflows/test_litert_models.yml?query=branch%3Amain)

* Tests that import, compile, and run
  [LiteRT (formerly TensorFlow Lite)](https://ai.google.dev/edge/litert) models
  through IREE then compare the outputs against a reference (`ai-edge-litert`).
* Runnable via [pytest](https://docs.pytest.org/).

### [onnx_models/](onnx_models/) : Open Neural Network Exchange models

[![Test ONNX Models](https://github.com/iree-org/iree-test-suites/actions/workflows/test_onnx_models.yml/badge.svg?branch=main)](https://github.com/iree-org/iree-test-suites/actions/workflows/test_onnx_models.yml?query=branch%3Amain)

* Tests that import, compile, and run ONNX models through IREE then compare
  the outputs against a reference (ONNX Runtime).
* Runnable via [pytest](https://docs.pytest.org/).

### [onnx_ops/](onnx_ops/) : Open Neural Network Exchange operations

[![Test ONNX Ops](https://github.com/iree-org/iree-test-suites/actions/workflows/test_onnx_ops.yml/badge.svg?branch=main)](https://github.com/iree-org/iree-test-suites/actions/workflows/test_onnx_ops.yml?query=branch%3Amain)

* 1000+ tests for [ONNX](https://onnx.ai/) framework
  [operators](https://onnx.ai/onnx/operators/).
* Runnable via [pytest](https://docs.pytest.org/) using a
  configurable set of flags to `iree-compile` and `iree-run-module`.

### [sharktank_models/](sharktank_models/) : Models exported using sharktank

[![Test Sharktank Models](https://github.com/iree-org/iree-test-suites/actions/workflows/test_sharktank_models.yml/badge.svg?branch=main)](https://github.com/iree-org/iree-test-suites/actions/workflows/test_sharktank_models.yml?query=branch%3Amain)

* Tests for small scale versions of Large Language Models (LLMs) and other
  Generative AI (GenAI) programs exported using the
  [sharktank package](https://github.com/nod-ai/shark-ai/tree/main/sharktank)
  built as part of the [shark-ai project](https://github.com/nod-ai/shark-ai).
* Runnable via [pytest](https://docs.pytest.org/) for both CPU and GPU targets.

### [torch_models/](torch_models/) : PyTorch models as Torch Dialect

[![Test Sharktank Models](https://github.com/iree-org/iree-test-suites/actions/workflows/test_torch_models.yml/badge.svg?branch=main)](https://github.com/iree-org/iree-test-suites/actions/workflows/test_torch_models.yml?query=branch%3Amain)

* Tests that compile and run PyTorch models, converted using torch-mlir.
* Runnable via [pytest](https://docs.pytest.org/) for any iree registered target.

## Git repository details

This repository uses [Git Large File Storage (LFS)](https://git-lfs.com/) to
store some model files (`.mlir`, `.mlirbc`) and parameters (`*.irpa`).

* If you would like to access these files, ensure you have `git-lfs` installed.
* If you have `git-lfs` installed and do _not_ want to download these files,
  you can set the `GIT_LFS_SKIP_SMUDGE=1` environment variable before cloning.

<!-- TODO: .lfsconfig file to make LFS default to not fetch?
    https://github.com/onnx/models?tab=readme-ov-file#usage
    https://github.com/onnx/models/blob/main/.lfsconfig
    https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-fetch.adoc -->
