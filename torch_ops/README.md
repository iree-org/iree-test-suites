# Torch Ops Test Suite

This is a test suite and test suite generation framework.
The goal of the test suite generation framework is to allow test writers to:
* Write torch programs that will then be used for differential testing.
* Use `numpy.allclose`'s when comparing floating point data.
* Deterministic generation of random data.
* Remove torch as a dependency from the test run.

There are several actions that one can do with this test suite and test suite generation framework.
* Running tests.
* Generate tests.
* Compute golden times (for benchmarks).

## Overview

Torch programs are exported to MLIR.
There is a test generator program that will take a torch module and a test generation configuration as an input to generate the following files.

```text
[module's class name]/[test case name]/
  test.mlir
  run_module_io.json
  expected_result_0.npy
  ...
```

Most of the test is specified within `run_module_io.json`, but usually one may want to run a test in multiple targets with different optimization levels and such.
To account for that, before running tests, one must supply a configuration json file with the following fields:

```text
config_name: Name of the configuration.
iree_compile_flags: Flags passed to iree-compile.
iree_run_module_flags: Flags passed to iree-run-module and similar tools.
```

Additionally, it may optionally include the following information:

```text
skip_compile_tests: List of tests which compilation should be skipped. Formatted as ${CLASS_DIR}/${TEST_DIR}
skip_run_tests: List of tests which run should be skipped.
expected_compile_failures: List of tests which one expects compilation to fail.
expected_run_failures: List of tests which one expects run to fail.
golden_times_ms: Dictionary of tests with golden times in ms.
```

With the generated configuration and the target configuration, one can run any test.

## Running tests

### Installation

Install the required packages for running tests.

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```
The test suite expects that you have `iree-base-compiler` and
`iree-base-runtime` packages installed or built from source

#### IREE: prebuilt wheels

```bash
pip install -r requirements-iree.txt
```

#### IREE: build from source

If you are using them from a source build, make sure that the iree python
bindings are discoverable in your `PYTHONPATH` and iree tools are in your
`PATH`.

```bash
export PATH="<path-to-iree-build>/tools:$PATH"
export PYTHONPATH="<path-to-iree-build>/bindings/python:$PYTHONPATH"
```

### Running Tests

- Run all tests 

```bash
pytest
```

- Run single test

```bash
pytest /point/to/test_directory/run_module_io.json
```

- Run benchmarking with rocprofv3

```
pytest --benchmark-with-rocprofv3
```

## Configuration files

### Generated configuration

Generated configuration files are written in JSON by the test generator tool.
Generated configuration files are named `run_module_io.json` by default.
They include things like:

* `function_name`: Value used in the `--function` flag when running iree-run-module. Defaults to "main".
* `flat_args`: Compressed arguments to be sent to the entry point. Defaults to None.
* `seed`: Seed used to generate the uncompressed arguments from the compressed arguments. Defaults to 0.
* `rtol`: Relative tolerance. Defaults to 1e-05.
* `atol`: Abolute tolerance. Defaults to 1e-08.
* `equal_nan`: Whether NaNs should compare as equal. Defaults to `False`.
* `mode`: Either "compare" or "benchmark". Defaults to "compare".
* `file_name`: The name of the mlir file. Defaults to "test.mlir".
* `vmfb_name`: The name given to the vmfb file. Defaults to "out.vmfb".
* `expected_output`: Only included when using "compare" mode. List of files holding expected returned values.

### Target configuration

Target configuration files are written in JSON
and allow one to specify different compilation and
run flags.

```json
{
  "config_name": "simple",
  "iree_compile_flags": [
    "--iree-hal-target-device=local",
    "--iree-hal-target-backends=llvm-cpu",
    "--iree-llvmcpu-target-cpu=host"
  ],
  "iree_run_module_flags": [],
  "skip_compile_tests": [],
  "skip_run_tests": [],
  "expected_compile_failures": [],
  "expected_run_failures": []
}
```

## Example Test Generation

To generate tests you will need to have iree-turbine
and pytorch installed.

```bash
pip install -r requirements-generate.txt
```

When writing new tests take into account the following guidelines.

```python
from generate_tests import gen, Formula, GenConfig

class AB(torch.nn.Module):
  def forward(self, A, B):
    return A @ B

def data():
  t = Formula(shape=(64, 64))
  yield GenConfig(**{"args": (t, t)})

gen(AB(), data())
```

or with a decorator style.

```python
from generate_tests import gen_tests, test, Formula, GenConfig

@gen_tests
class AB(torch.nn.Module):
  def forward(self, A, B):
    return A @ B

  @staticmethod
  @test
  def test_data():
    t = Formula(shape=(64, 64))
    yield GenConfig(**{"args": (t, t)})

AB()
```

The GenConfig class is a dataclass that holds all necessary information for generating and running the test.
Fields of the GenConfig class are documented with its definition.
The most common ones are below:

* `name`: Name of the test. Will be used when creating the test folder directory.
* `args`: Example arguments to aot.export which will also be used as real inputs for the tests. These are in compressed form.
* `kwargs`: Example kwargs to aot.export which will also be used as real inputs for the tests. These are in compresed form.
* `dynamic_shapes`: Dynamic shape specification.
* `seed`: Seed used to generate inputs from compressed arguments.
* `rtol`: Relative tolerance.
* `atol`: Absolute tolerance.
* `mode`: "benchmark" or "compare".

### Formulas / Compressed Arguments

Since there may be many input tensors taking a large quantity of memory, input tensors are "compressed" into Formulas.
A Formula represents the a random tensor of the following form:

```python
def formula(shape, coeff, offset, dtype):
  return (coeff * numpy.random.rand(*shape) + offset).astype(dtype)
```

These Formulas will generate the concrete tensors during test generation and testing using the same seed provided by GenConfig.
The seed is done on a per-test level as opposed to a per-Formula level.
This is because we want tests to have deterministic random inputs and we only need to set the seed once per test to achieve this.
Otherwise, one would need to type a different seed for every different formula.
