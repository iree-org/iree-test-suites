## Regression tests

### Adding your own model

- To add your own model, create a directory under `quality_tests` and add JSON files that correspond to the submodels. Please follow the [JSON file schema in this README file](#required-and-optional-fields-for-the-json-model-file)

### How to run

- Example command to run quality tests for a specific model

```
python sharktank_models/quality_tests/ \
    -rpFe \
    --log-cli-level=info \
    --timeout=600 \
    --durations=0
```

### Required and optional fields for the JSON model file

| Field Name                          | Required | Type    | Description                                                                                                                                      |
| ----------------------------------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| inputs                              | optional | array   | An array of objects that provides the input blob and the expected input value (ex: `{"source" :"", "value": ""}`, the value field is optional)   |
| outputs                             | optional | array   | An array of objects that provides the output blob and the expected output value (ex: `{"source" :"", "value": ""}`, the value field is optional) |
| real_weights                        | optional | string  | URL that provides the real weights blob                                                                                                          |
| mlir                                | required | string  | URL that provides the MLIR blob                                                                                                                  |
| pipeline_mlir                       | optional | string  | URL that provides the MLIR blob                                                                                                                  |
| device                              | optional | string  | The device to run the threshold tests on                                                                                                                  |
| compiler_flags                      | optional | array   | Compiler flag options for the iree compilation                                                                                               |
| pipeline_compiler_flags             | optional | array   | Compiler flag options for the pipeline iree compilation                                                                                     |
| threshold_args                      | optional | array   | The expected threshold value for `iree_run_module` to indicate if the test passed or not , ex: `["--expected_f16_threshold=1.0f"]`           |
| run_function                    | optional | string  | The function that the `iree_run_module` in the threshold tests                                                                               |
| run_test_expecting_to_fail      | optional | boolean | If true, the threshold test will expect to fail                                                                                              |
| compile_chip_expecting_to_fail | optional | array   | If an array is passed in, the compilation tests will fail on the specified chip, ex: `["gfx90a"]`                                           |
| compile_only                        | optional | boolean | If true, only the compilation tests will run                                                                                                     |
| add_pipeline_module                 | optional | boolean | If true, the <b>pipeline mlir</b> module will be added to the `iree_run_module` as an argument                                                   |
| tuner_file                          | optional | dict    | Adds a `iree-codegen-transform-dialect-library` compiler flag for a SKU-specific tuner file (ex: `{"mi308": "{path_to_tuner_file}"}`)       |

Please feel free to look at any JSON examples under a model directory (ex: sd3, sdxl)
