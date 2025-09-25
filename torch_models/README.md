# Torch Models Test Suite

## Example Usage

```base
pytest -m "hip" --test-file-directory=./sdxl --module-directory=modules --external-file-directory=./test_suite_files  --log-level=INFO -o log_cli=True --force-recompile=True
```

Explanation of flags:

```
pytest \
-m "cpu" # Run only cpu tests
--test-file-directory=./sdxl # Directory containing test JSON files
--module-directory=modules # Directory containing module JSON files
--external-file-directory=./test_suite_files # Directory containing any external files referenced by the JSON files
--log-level=INFO # Set logging level to INFO
-o log_cli=True # Enable live logging output during test runs
--force-recompile=True # Force recompilation of modules even if a cached compiled module already exists from a previous run
```

## Using pytest to control collection

The test suite is designed to work well with pytest. Some useful commands:

### Markers

- `pytest -m benchmark` to run only benchmark tests.
- `pytest -m quality` to run only quality tests.
- `pytest -m "quality and cpu"` to run only quality tests on cpu.
- `pytest -m "benchmark or gfx1201"` to run benchmark tests or tests on gfx1201.

You can pass the `--collect-only` flag to see which tests would be ran without
running them.

See
https://python-basics-tutorial.readthedocs.io/en/24.3.0/test/pytest/markers.html
for more information on pytest markers.

### Logging

To see logging output, use `--log-level=INFO`.

To see live logging output during test runs, use the `-o log_cli=true` flag:

### Flags

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

## Module Definitions

Please feel free to look at any JSON examples under the modules directory for reference.

| Field Name                     | Required | Type    | Description                                                                                                                                      |
| ------------------------------ | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| mlir                           | required | string  | URL that provides the MLIR blob                                                                                                                  |
| compiler_flags                 | required | array   | Compiler flag options for the iree compilation                                                                                                   |

## Test Definitions

Please feel free to look at any JSON examples under the sdxl/ directory for reference.

### Common Fields

| Field Name                     | Required | Type    | Description                                                                                                                                      |
| ------------------------------ | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| type                           | required | string  | The type of test definition. Example: `benchmark`, `quality`. Also acts as a pytest marker.                                                      |
| markers                        | required | array   | List of pytest markers to apply to the test.                                                                                                     |
| modules                        | required | array   | List of modules required for this test.                                                                                                          |
| weights                        | optional | array   | List of weights to use for this test. Each weight definition contains a scope and a url.                                                         |
| inputs                         | required | argspec | List of inputs to use for this test.                                                                                                             |
| outputs                        | required | argspec | List of outputs to use for this test.                                                                                                            |
| expected_outputs               | required | argspec | List of expected outputs for this test.                                                                                                          |
| run_args                       | optional | array   | Additional runtime arguments to pass to the iree-run-module/iree-benchmark-module command.                                                       |

`argspec` is defined as:

```
{
    "url": "<url to file>",
    "value": "<literal value> | <file path> | <byte string>"
}
All fields are optional, but at least one must be present.
The resulting argument string is a concatenation of the fields:
  <value>=@<url_file_path>

For example:
{
   "url": "https://example.com/data.txt",
   "value": 1xf16
}

would be passed (say as an input) as:
  --input=1xf16=@/path/to/downloaded/data.txt
```

### Quality Test Specific Fields

### Benchmark Test Specific Fields

| Field Name                     | Required | Type    | Description                                                                                                                                      |
| ------------------------------ | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| golden_time                    | optional | float   | golden time in ms                                                                                                                                                 |

## TODO

- Add splat weight support.
- Add random weight generated support.
- Add binary size / dispatch size support.
