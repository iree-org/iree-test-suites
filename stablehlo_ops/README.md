# StableHLO Ops Test Suite

This suite contains MLIR snippets for most StableHLO operations. Each test
passes the same inputs to the static and dynamic versions of an operation.
The StableHLO interpreter runs the static version once to compute the expected
output tensors for both versions.

## Overview

Tests are written by hand. The project has the following layout.

```text
cases/
  add/
    run_module_io.json
    static.mlir
    dynamic.mlir
  and/
    run_module_io.json
    static.mlir
    dynamic.mlir

tensors/
  1xf32_a.json
  10xf32_a.json
  10xf32_b.json

test_driver/
  __init__.py
  runner.py
  ...

configs/
  stablehlo_ops_cpu_llvm_sync.json
```

- `cases` groups tests by operation. Each `run_module_io.json` file connects the MLIR modules to their inputs.
- `tensors` holds the shared input tensors.
- `test_driver` contains the pytest plugin that collects and runs the cases. It runs `iree-compile` and `iree-run-module` on both variants and uses the interpreter on the static variant to compute the reference outputs.
- `configs` contains flags for `iree-compile` and `iree-run-module`, along with expected test failures.

## Installation

The pinned StableHLO interpreter wheel requires CPython 3.11 on Linux x86_64.
Install the base requirements from the `stablehlo_ops/` directory.

```bash
python3.11 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

The tests require IREE. You can install the prebuilt wheels or build IREE from source.

### Using prebuilt IREE wheels

```bash
pip install -r requirements-iree.txt
```

### Using an IREE build from source

After a successful build, add the IREE tools directory to your `PATH`.

```bash
export PATH="<path-to-iree-build>/tools:$PATH"
```

## Running tests

Run all tests.

```bash
pytest
```

Run the cases in parallel with four pytest workers.

```bash
pytest -n 4
```

Test a single operation, such as `abs`.

```bash
pytest cases/abs/
```

## Add a case

1. Write a module, such as `cases/abs/static.mlir`:

   ```mlir
   module {
     func.func @main(%a: tensor<10xf32>) -> tensor<10xf32> {
       %result = stablehlo.abs %a : tensor<10xf32>
       return %result : tensor<10xf32>
     }
   }
   ```

2. Write the corresponding dynamic module at `cases/abs/dynamic.mlir`
   and add one test description at `cases/abs/run_module_io.json`:

   ```json
   {
     "file_name": "static.mlir",
     "dynamic_file_name": "dynamic.mlir",
     "inputs": [
       "../../tensors/10xf32_a.json"
     ]
   }
   ```

All paths are relative to the manifest. Inputs are ordered function arguments.
Each module must define its entry point as `@main`.

Tensor fixtures are stored as readable JSON. Each file specifies the NumPy dtype, shape, and values.

```json
{
  "dtype": "float32",
  "shape": [3],
  "values": [1.5, -2.0, 0.25]
}
```

The runner compares integer and boolean outputs exactly. Floating-point outputs
use suite-wide tolerances of `rtol=1e-5` and `atol=1e-8`. NaNs compare unequal.
The StableHLO interpreter computes the expected tensors, so the manifest needs no
`expected_output` paths.

When a case supplies `stablehlo_interpreter_fallback.mlir` beside its manifest,
the interpreter evaluates that module with the same inputs.
