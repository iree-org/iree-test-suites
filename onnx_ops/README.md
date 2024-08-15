# ONNX Operator Tests

This test suite exercises ONNX (Open Neural Network Exchange: https://onnx.ai/)
operators (https://onnx.ai/onnx/operators/).

Testing follows several stages:

```mermaid
graph LR
  Import -. "\n(offline)" .-> Compile
  Compile --> Run
```

Importing is run "offline" and the outputs are checked in to the repository for
ease of use in downstream projects and by developers who prefer to work directly
with `.mlir` files and native (C/C++) tools.

## Quickstart

1. Set up your virtual environment and install requirements:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    ```

    * To use `iree-compile` and `iree-run-module` from Python packages:

        ```bash
        python -m pip install -r requirements-iree.txt
        ```

    * To use local versions of `iree-compile` and `iree-run-module`, put them on
      your `$PATH` ahead of your `.venv/Scripts` directory:

        ```bash
        export PATH=path/to/iree-build:$PATH
        ```

2. Run pytest using typical flags:

    ```bash
    pytest \
      -n auto \
      -rA \
      --timeout=30 \
      --durations=20 \
      --config-files=configs/onnx_ops_cpu_llvm_sync.json \
      --report-log=/tmp/onnx_ops_cpu_logs.json
    ```

    See https://docs.pytest.org/en/stable/how-to/usage.html for other options.

## Test case structure

Each test case is a folder containing a few files:

```text
[test case name]/
  model.mlir
  input_0.bin
  input_1.bin
  ...
  output_0.bin
  output_1.bin
  ...
  run_module_io_flags.txt
```

Where:

* `model.mlir` is in a format that is ready for use with `iree-compile`:

    ```mlir
    module {
      func.func @test_add(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
        %none = torch.constant.none
        %0 = torch.operator "onnx.Add"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
        return %0 : !torch.vtensor<[3,4,5],f32>
      }
    }
    ```

* `input_0.bin` and `output_0.bin` files correspond to any number of program
  inputs and outputs for one test case
* `run_module_io_flags.txt` is a flagfile for use with
  `iree-run-module --flagfile=run_module_io_flags.txt` of the format:

  ```text
  --input=2x3xf32=@input_0.bin
  --expected_output=2x3xf32=@output_0.bin
  ```

## Running tests

Tests are run using the [pytest](https://docs.pytest.org/en/stable/) framework.

A [`conftest.py`](conftest.py) file collects test cases from subdirectories,
wrapping each directory matching the format described above to one test case
per test configuration. Test configurations are defined in JSON config files
like [`configs/onnx_ops_cpu_llvm_sync.json`](./configs/onnx_ops_cpu_llvm_sync.json).

### Updating expected failure lists

Each config file uses with pytest includes a list of expected compile and run
failures like this:

```json
  "expected_compile_failures": [
    "test_acos",
  ],
  "expected_run_failures": [
    "test_add_uint8",
  ],
```

To update these lists using the results of a test run:

1. Run pytest with the `--report-log` option:

    ```bash
    pytest \
      --report-log=/tmp/onnx_ops_cpu_logs.json \
      --config-files=onnx_ops_cpu_llvm_sync.json \
      ...
    ```

2. Run the `update_config_xfails.py` script:

    ```bash
    python update_config_xfails.py \
      --log-file=/tmp/onnx_ops_cpu_logs.json \
      --config-file=onnx_ops_cpu_llvm_sync.json
    ```

You can also update the config JSON files manually. The log output on its own
should give enough information for each test case (e.g.
"remove from 'expected_run_failures'" for newly passing tests), but there can be
1000+ test cases, so the automation can save time.

### Advanced pytest usage

* The `--ignore-xfails` option will ignore any expected compile or runtime
  failures.
* The `--skip-all-runs` option will only `iree-compile` tests, not
  `iree-run-module` tests.

## Generating tests

Test cases are imported from upstream ONNX tests:

Directory in [onnx/onnx](https://github.com/onnx/onnx/) | Description
-- | --
[`onnx/backend/test/case/`](https://github.com/onnx/onnx/tree/main/onnx/backend/test/case) | Python source files
[`onnx/backend/test/data/`](https://github.com/onnx/onnx/tree/main/onnx/backend/test/data) | Generated `.onnx` and `[input,output]_[0-9]+.pb` files

The [`import_onnx_tests.py`](./onnx/import_onnx_tests.py) script walks the
`data/` folder and generates test cases into our local
[`generated/` folder](./generated/).

To regenerate the test cases:

```bash
# Virtual environment setup.
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

# Import all test cases (may take a few minutes).
python import_onnx_tests.py
```
