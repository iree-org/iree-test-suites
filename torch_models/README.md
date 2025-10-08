# Torch Models Test Suite

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

## Example Usages

- Run all tests under the directory `./examples` with the module definitions
  relative to `./`:

```bash
pytest \
    --test-file-directory=./examples \
    --module-directory=./ \
    --external-file-directory=./ \
    --log-cli-level=info
```

- Run all tests marked as `hip` and `benchmark`:

```bash
pytest \
    --test-file-directory=./examples \
    --module-directory=./ \
    --external-file-directory=./ \
    --log-cli-level=info \
    -m "hip and benchmark"
```

See
https://python-basics-tutorial.readthedocs.io/en/24.3.0/test/pytest/markers.html
for more information on pytest markers.

- Run all tests marked as `hip` and `benchmark` with live logging disabled:

```bash
pytest \
    --test-file-directory=./examples \
    --module-directory=./ \
    --external-file-directory=./ \
    -m "hip and benchmark" \
```

- Collect all `hip` or `cpu` tests without running them:

```bash
pytest \
    --test-file-directory=./examples \
    --module-directory=./ \
    --external-file-directory=./ \
    --log-cli-level=info \
    -m "hip or cpu" \
    --collect-only
```

- Run all tests under an external directory with cwd as `iree-test-suites`:

```bash
pytest torch_modles \
    --test-file-directory=<path-to-iree>/tests/external/iree-test-suites/torch_models \
    --module-directory=<path-to-iree>/tests/external/iree-test-suites/torch_models \
    --external-file-directory=<path-to-iree>/tests/external/iree-test-suites/test_suite_files \
    --log-cli-level=info
```

- Force recompilation of modules even if modules are cached from a previous run:

```bash
pytest \
    --test-file-directory=./examples \
    --module-directory=./ \
    --external-file-directory=./ \
    --log-cli-level=info \
    -m "hip or cpu" \
    --force-recompile
```

### pytest-iree Flags

- `--test-file-directory`: The directory containing the test JSON files.
- `--module-directory`: The directory containing the module JSON files. All
  module paths are defined relative to this directory.
- `--external-file-directory`: The directory containing any external files
  referenced by the test JSON files. All file paths (other than modules) are
  defined relative to this directory.

- `--artifact-directory`: The directory to store any artifacts (e.g. compiled
  modules). Defaults to `./artifacts` in the current working directory.
- `--force-recompile`: If set, forces recompilation of modules even if a cached
  compiled module already exists. Defaults to False. Useful for testing
  compiler changes.

## Tips

- By default, the plugin caches all compiled modules and downloaded artifacts.
  If you are doing compiler development, you always want to set the
  `--force-recompile` flag. If you are building tests, but not modifying module
  definitions, you should keep it on to not have compilation overhead.
- The caching of modules and downloaded artifacts is predictable. Look at the
  artifact class definition to find out how the particular artifact is cached.
- Every iree command is ran with "external-file-directory" as the cwd. So if
  you have any relative paths in your module definitions or test definitions,
  they are relative to that directory.

## Module Definitions

Please feel free to look at any JSON examples under the `examples/modules` directory for reference.

| Field Name                     | Required | Type    | Description                                                                                                                                      |
| ------------------------------ | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| mlir                           | required | string  | URL that provides the MLIR blob                                                                                                                  |
| compiler_flags                 | required | array   | Compiler flag options for the iree compilation                                                                                                   |

## Test Definitions

Please feel free to look at any JSON examples under the `examples` directory for reference.

### Common Fields

### Quality Test Definition

| Field Name                     | Required | Type    | Description                                                                                                                                      |
| ------------------------------ | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| type                           | required | string  | The type of test definition. Must be `quality`. Also acts as a pytest marker.                                                                    |
| markers                        | required | array   | List of pytest markers to apply to the test.                                                                                                     |
| modules                        | required | array   | List of modules required for this test.                                                                                                          |
| weights                        | optional | array   | List of weights to use for this test. Each weight definition is a `weightspec`.                                                                  |
| inputs                         | required | argspec | List of inputs to use for this test.                                                                                                             |
| expected_outputs               | required | argspec | List of expected outputs for this test.                                                                                                          |
| run_args                       | optional | array   | Additional runtime arguments to pass to the iree-run-module/iree-benchmark-module command.                                                       |

### Benchmark Test Definition

| Field Name                     | Required | Type    | Description                                                                                                                                      |
| ------------------------------ | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| type                           | required | string  | The type of test definition. Must be `benchmark`. Also acts as a pytest marker.                                                                  |
| markers                        | required | array   | List of pytest markers to apply to the test.                                                                                                     |
| modules                        | required | array   | List of modules required for this test.                                                                                                          |
| weights                        | optional | array   | List of weights to use for this test. Each weight definition is a `weightspec`.                                                                  |
| inputs                         | required | argspec | List of inputs to use for this test.                                                                                                             |
| run_args                       | optional | array   | Additional runtime arguments to pass to the iree-run-module/iree-benchmark-module command.                                                       |
| golden_time_ms                 | optional | float   | golden time in ms                                                                                                                                |
###  Compstat Test Definition

| Field Name                     | Required | Type    | Description                                                                                                                                      |
| ------------------------------ | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| type                           | required | string  | The type of test definition. Must be `compstat`. Also acts as a pytest marker.                                                                   |
| markers                        | required | array   | List of pytest markers to apply to the test.                                                                                                     |
| module                         | required | array   | Module for which this test is defined.                                                                                                           |
| golden_dispatch_count          | optional | float   | Maximum number of dispatch count allowed.                                                                                                        |
| golden_binary_size             | optional | float   | Maximum binary size allowed.                                                                                                                     |

### Custom Argument Specifications

#### weightspec

`weightspec` is defined as:

```
{
    "type": "url" | "random"
    "scope": "<string>"

    # Only for type=url
    "url": "<url to file>", # Only for type=url

    # Only for type=random
    "module": "<module-name>"
    "seed": <int>
}
```

For example:
```
{
   "type": "url",
   "url": "https://example.com/weights.pt",
   "scope": "model"
}
```

```
{
    "type": "random",
    "module": "examples/modules/scheduled_unet_gfx1201"
    "seed": 42,
    "scope": "model"
}
```

#### argspec

`argspec` is defined as:

```
{
    "url": "<url to file>",
    "value": "<literal value> | <file path> | <byte string>"
}
```
All fields are optional, but at least one must be present.
The resulting argument string is a concatenation of the fields:
  `<value>=@<url_file_path>`

For example:
```
{
   "url": "https://example.com/data.txt",
   "value": 1xf16
}
```

would be passed (say as an input) as:
  `--input=1xf16=@/path/to/downloaded/data.txt`

## TODO

- Add splat weight support.
- Add random weight generated support.
