## Regression tests

### Adding your own model

- To add your own model, create a directory under `regression_tests` and add JSON files that correspond to the submodels. Please follow the [JSON file schema in this README file](#required-and-optional-fields-for-the-json-model-file)

### How to run

- Example command to run quality tests for a specific model

```
python sharktank_models/regression_tests/run_quality_tests.py --model=sdxl --submodel=*

python sharktank_models/regression_tests/run_quality_tests.py --model=sdxl --submodel=clip
```

Argument options for the script

| Argument Name | Default value | Description                                                                                                                                      |
| ------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| --model       | sdxl          | Runs quality tests for a specific model                                                                                                        |
| --submodel    | \*            | If specified, the quality tests will run for a specific submodel (ex: `--submodel clip`). If not specified, it will run tests on all submodels |
| --sku         | mi300         | The quality tests will run on this sku and retrieve golden values from the specified sku                                                       |
| --backend   | gfx942        | The quality tests will run on this backend                                                                                                   |

### Required and optional fields for the JSON model file

| Field Name                          | Required | Type    | Description                                                                                                                                      |
| ----------------------------------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| inputs                              | optional | array   | An array of objects that provides the input blob and the expected input value (ex: `{"source" :"", "value": ""}`, the value field is optional)   |
| outputs                             | optional | array   | An array of objects that provides the output blob and the expected output value (ex: `{"source" :"", "value": ""}`, the value field is optional) |
| real_weights                        | optional | string  | URL that provides the real weights blob                                                                                                          |
| mlir                                | required | string  | URL that provides the MLIR blob                                                                                                                  |
| pipeline_mlir                       | optional | string  | URL that provides the MLIR blob                                                                                                                  |
| cpu_compiler_flags                  | optional | array   | Compiler flag options for the CPU iree compilation                                                                                               |
| rocm_compiler_flags                 | optional | array   | Compiler flag options for the ROCM iree compilation                                                                                              |
| rocm_pipeline_compiler_flags        | optional | array   | Compiler flag options for the ROCM pipeline iree compilation                                                                                     |
| cpu_threshold_args                  | optional | array   | The expected threshold CPU value for `iree_run_module` to indicate if the test passed or not , ex: `["--expected_f16_threshold=1.0f"]`           |
| rocm_threshold_args                 | optional | array   | The expected threshold ROCM value for `iree_run_module` to indicate if the test passed or not , ex: `["--expected_f16_threshold=1.0f"]`          |
| run_cpu_function                    | optional | string  | The function that the `iree_run_module` in the CPU threshold tests                                                                               |
| run_rocm_function                   | optional | string  | The function that the `iree_run_module` in the ROCM threshold tests                                                                              |
| cpu_run_test_expecting_to_fail      | optional | boolean | If true, the CPU threshold test will expect to fail                                                                                              |
| rocm_run_test_expecting_to_fail     | optional | boolean | If true, the ROCM threshold test will expect to fail                                                                                             |
| rocm_tests_only                     | optional | boolean | If true, only the ROCM compilation and threshold test will run                                                                                   |
| rocm_compile_chip_expecting_to_fail | optional | array   | If an array is passed in, the ROCM compilation tests will fail on the specified chip, ex: `["gfx90a"]`                                           |
| compile_only                        | optional | boolean | If true, only the compilation tests will run                                                                                                     |
| add_pipeline_module                 | optional | boolean | If true, the <b>pipeline mlir</b> module will be added to the `iree_run_module` as an argument                                                   |
| tuner_file                          | optional | dict    | Adds a `iree-codegen-transform-dialect-library `ROCM compiler flag for a SKU-specific tuner file (ex: `{"mi308": "{path_to_tuner_file}"}`)       |

Please feel free to look at any JSON examples under a model directory (ex: sd3, sdxl)
