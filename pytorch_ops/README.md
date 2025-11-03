# Torch Ops Test Suite

## Installation

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```
The test suite expects that you have `iree-base-compiler` and
`iree-base-runtime` packages installed or built from source

### IREE: prebuilt wheels

```bash
pip install -r requirements-iree.txt
```

### IREE: build from source

If you are using them from a source build, make sure that the iree python
bindings are discoverable in your `PYTHONPATH` and iree tools are in your
`PATH`.

```bash
export PATH="<path-to-iree-build>/tools:$PATH"
export PYTHONPATH="<path-to-iree-build>/bindings/python:$PYTHONPATH"
```

## Running Tests

- Run all tests 

```bash
pytest
```

- Run single test

```bash
pytest test_a_b
```

## Configuration file

Configuration files are written in JSON
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
# Instead of using torch.nn.Module
# as a base class for a torch Module,
# use this class instead.
#             V
class MyTest(TestGenerator):

  def forward(self, left, right):
    return left @ right

  # Use the test decorator to denote a test.
  # V
  @test
    # The output is a tuple of (Sequence, dict)
    # that corresponds to *args, **kwargs
    #                       V
    def test_16x16(self) -> tuple[Sequence, dict]:
      # Use self.rand instead of torch.rand.
      # It is a wrapper around torch.rand to make sure tests are deterministic
      # and seeds are easily specified
      #         V                  V
      return ((self.rand(16, 16), self.rand(16, 16), {})

    # Add more tests by adding a new decorator
    @test(seed=42, atol=1e-7, rtol=1e-9)
    def test_8x8(self):
      return ((self.rand(8, 8), self.rand(8, 8), {})

# Will create a directory with the test and metadata needed to execute
# the test.
MyTest().generate_tests()
```

## Test case structure

Each test case is a folder containing a few files:

```text
[test case name]/
  test.mlir
  input0.npy
  input1.npy
  ...
  expected_output0.npy
  expected_output1.npy
  ...
  run_module_io_flags.json
```

* `test.mlir` is in torch-mlir dialect.
* `input0.npy` and `expected_output.npy` files correspond to any of number of program
  inputs and outputs for one test case
* `run_module_io_flags.json` is a json file similar the flagfile.
  However, this file is a json file and supports additional metadata.
  For example,
    * `rtol` is relative tolerance,
    * `atol` is absolute tolerance, and
    * `equal_nan` which correpond to the parameters given to `numpy.allclose`.
    * Additionally, `ovserved_outputs` corresponds to the option passed to the `output` flag
      when using iree-run-module.

```json
{
    "rtol": 1e-05,
    "atol": 1e-08,
    "equal_nan": false,
    "inputs": [
        "input0.npy",
        "input1.npy"
    ],
    "expected_outputs": [
        "expected_result0.npy"
    ],
    "observed_outputs": [
        "observed_result0.npy"
    ]
}
```

### Advanced pytest usage

* The `--ignore-xfails` option will ignore any expected compile or runtime
  failures.
* The `--skip-all-runs` option will only `iree-compile` tests, not
  `iree-run-module` tests.
