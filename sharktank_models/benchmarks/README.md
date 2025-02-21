## Benchmark tests

### Adding your own model

- To add your own model, create a directory under `benchmarks` and add JSON files that correspond to the submodels and chip. Please follow the [JSON file schema in this README file](#required-and-optional-fields-for-the-json-model-file)

### How to run

```
python sharktank_models/benchmarks/run_benchmarks.py --model=sdxl --filename=*

python sharktank_models/benchmarks/run_benchmarks.py --model=sdxl --filename=clip_rocm
```

Argument options for the script

| Argument Name | Default value | Description                                                                                                                                      |
| ------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| --model       | sdxl          | Runs benchmark tests for a specific model                                                                                                        |
| --filename    | \*            | If specified, the benchmark tests will run for a specific filename (ex: `--filename clip`). If not specified, it will run tests on all filenames |
| --sku         | mi300         | The benchmark tests will run on this sku and retrieve golden values from the specified sku                                                       |
| --backend   | gfx942        | The benchmark tests will run on this backend                                                                                                   |

### Required and optional fields for the JSON model file

| Field Name                       | Required | Type    | Description                                                                                                                  |
| -------------------------------- | -------- | ------- | ---------------------------------------------------------------------------------------------------------------------------- |
| inputs                           | required | array   | An array of input strings for the benchmark module (ex: `["1xi64, 1xf16]`)                                                   |
| compilation_required             | optional | boolean | If true, this will let the benchmark test know that it needs to compile a file                                               |
| compiled_file_name               | optional | string  | When the compilation occurs, this will be the file name                                                                      |
| compile_flags                    | optional | array   | An array of compiler flag options                                                                                            |
| mlir_file_path                   | optional | string  | Path to where the mlir file to compile is                                                                                    |
| modules                          | optional | array   | Specific to e2e, add modules here to include in the benchmarking test                                                        |
| function_run                     | required | string  | The function that the `iree-benchmark-module` will run adnd benchmark                                                        |
| benchmark_repetitions            | required | float   | The number of times the benchmark tests will repeat                                                                          |
| benchmark_min_warmup_time        | required | float   | The minimum warm up time for the benchmark test                                                                              |
| device                           | required | string  | The device that the benchmark tests are running                                                                              |
| golden_time_tolerance_multiplier | optional | object  | An object of tolerance multipliers, where the key is the sku and the value is the multiplier, (ex: `{"mi250": 1.3}`)         |
| golden_time_ms                   | optional | object  | An object of golden times, where the key is the sku and the value is the golden time in ms, (ex: `{"mi250": 100}`)           |
| golden_dispatch                  | optional | object  | An object of golden dispatches, where the key is the sku and the value is the golden dispatch count, (ex: `{"mi250": 1602}`) |
| golden_size                      | optional | object  | An object of golden sizes, where the key is the sku and the value is the golden size in bytes, (ex: `{"mi250": 2000000}`)    |
| specific_chip_to_ignore     | optional | array   | An array of chip values, where the benchmark tests will ignore the chips specified                                           |
| real_weights_file_name           | optional | string  | If real weights is a different file name, specify it here in order to get the correct real weights file                      |

Please feel free to look at any JSON examples under a model directory (ex: sdxl)
