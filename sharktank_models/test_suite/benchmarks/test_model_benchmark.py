# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from collections import namedtuple
import logging
from typing import Sequence
import subprocess
import json
from pathlib import Path
import tabulate
from ireers_tools import *
from pytest_check import check
import pytest

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent
vmfb_dir = os.getenv("TEST_OUTPUT_ARTIFACTS", default=str(THIS_DIR))
artifacts_dir = f"{os.getenv('IREE_TEST_FILES', default=str(THIS_DIR))}/artifacts"
artifacts_dir = Path(os.path.expanduser(artifacts_dir)).resolve()
rocm_chip = os.getenv("ROCM_CHIP", default="gfx942")
sku = os.getenv("SKU", default="mi300")
model_name = os.getenv("THRESHOLD_MODEL", default="sdxl")
submodel_name = os.getenv("THRESHOLD_SUBMODEL", default="*")

SUBMODEL_FOLDER_PATH = THIS_DIR / f"{model_name}"

# if a specific submodel in the environment variable is not specified, all the submodels under the model directory will be tested
parameters = []
if submodel_name != "*":
    parameters = [submodel_name]
else:
    for filename in os.listdir(SUBMODEL_FOLDER_PATH):
        if ".json" in filename:
            parameters.append(filename.split(".")[0])

"""
Helper methods
"""
# runs an iree command using subprocess
def run_iree_command(args: Sequence[str] = ()):
    command = "Exec:", " ".join(args)
    logger.info(command)
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (stdout_v, stderr_v) = (proc.stdout, proc.stderr)
    return_code = proc.returncode
    if return_code == 0:
        return 0, proc.stdout
    logger.error(
        f"Command failed!\n"
        f"Stderr diagnostics:\n{proc.stderr}\n"
        f"Stdout diagnostics:\n{proc.stdout}\n"
    )
    return 1, proc.stdout

# Converts a list of inputs into compiler friendly input arguments
def get_input_list(input_list):
    return [f"--input={entry}" for entry in input_list]

# Parses through results and returns the mean benchmark time
def job_summary_process(ret_value, output, model_name):
    if ret_value == 1:
        # Output should have already been logged earlier.
        pytest.fail(f"Running {model_name} benchmark failed. Exiting.")
    
    output_json = json.loads(output)
    benchmark_mean_time = decode_output(output_json)
    if benchmark_mean_time == -1:
        pytest.fail(f"Benchmark results was not found for {model_name} during benchmark module run. Exiting.")
    
    return benchmark_mean_time

# Decodes the output of the benchmark results
def decode_output(bench_lines):
    for line in bench_lines.get("benchmarks"):
        if line.get("aggregate_name") == "mean":
            return float(line.get("real_time"))
        
    return -1

# iree-compile helper method, allowing custom file path, compile flags and output compiled file name
def compile_iree_method(mlir_file_path, compile_flags, compiled_file_name):
    # Adding all the compiler arguments together
    MLIR_COMPLETE_FILE_PATH = str(THIS_DIR / mlir_file_path)
    exec_args = [
        "iree-compile",
        f"{MLIR_COMPLETE_FILE_PATH}"
        "--iree-hal-target-backends=rocm",
        f"--iree-hip-target={rocm_chip}",
    ] + compile_flags + [
        "-o",
        f"{vmfb_dir}/{compiled_file_name}_rocm.vmfb"
    ]
    ret_value, stdout = run_iree_command(exec_args)
    return ret_value, stdout

# Specific to end to end tests, adding initial benchmark arguments and custom modules
def e2e_iree_benchmark_module_args(modules, compiled_file_name):
    exec_args = [
        "iree-benchmark-module",
        f"--device=hip",
        "--device_allocator=caching"
    ]
    
    # for e2e tests, we are adding the submodel modules 
    for module in modules:
        exec_args.append(f"--module={vmfb_dir}/{module}_vmfbs/model.rocm_{rocm_chip}.vmfb")
        exec_args.append(f"--parameters=model={artifacts_dir}/{module}/real_weights.irpa")
    # adding the full e2e pipeline module
    exec_args.append(f"--module={vmfb_dir}/{compiled_file_name}_rocm.vmfb")
    return exec_args


@pytest.mark.parametrize("submodel_name", parameters)
class TestModelBenchmark:
    @pytest.fixture(autouse = True)
    @classmethod
    def setup_class(self, submodel_name):
        self.model_name = model_name
        self.submodel_name = submodel_name

        SUBMODEL_FILE_PATH = THIS_DIR / f"{model_name}/{self.submodel_name}.json"

        with open(SUBMODEL_FILE_PATH, 'r') as file:
            data = json.load(file)

            self.inputs = data.get("inputs", [])
            self.function_run = data.get("function_run")
            self.benchmark_repetitions = data.get("benchmark_repetitions")
            self.benchmark_min_warmup_time = data.get("benchmark_min_warmup_time")
            
            # retrieving golden values
            self.golden_time_tolerance_multiplier = data.get("golden_time_tolerance_multiplier", {}).get(sku)
            self.golden_time = data.get("golden_time_ms", {}).get(sku)
            self.golden_dispatch = data.get("golden_dispatch", {}).get(sku)
            self.golden_size = data.get("golden_size", {}).get(sku)
            
            # Custom configurations
            self.specific_rocm_chip_to_ignore = data.get("specific_rocm_chip_to_ignore", [])
            self.real_weights_file_name = data.get("real_weights_file_name", "real_weights.irpa")

            # custom configurations related to e2e testing
            self.compilation_required = data.get("compilation_required", False)
            self.compiled_file_name = data.get("compiled_file_name")
            self.compile_flags = data.get("compile_flags", [])
            self.mlir_file_path = data.get("mlir_file_path", "")
            self.modules = data.get("modules", [])


    def test_rocm_benchmark(self):
        # if a chip is designated to be ignored in JSON file, skip test
        if rocm_chip in self.specific_rocm_chip_to_ignore:
            pytest.skip(f"Ignoring benchmark test for {self.model_name} {self.submodel_name} for chip {rocm_chip}")

        # if compilation is required, run this step
        if self.compilation_required:
            ret_value, stdout = compile_iree_method(self.mlir_file_path, self.compile_flags, self.compiled_file_name)
            if ret_value == 1:
                return 1, stdout
            
        directory_compile = f"{vmfb_dir}/{self.model_name}_{self.submodel_name}_vmfbs"
        directory = f"{artifacts_dir}/{self.model_name}_{self.submodel_name}"

        exec_args = [
            "iree-benchmark-module",
            f"--device=hip",
            "--device_allocator=caching",
            f"--module={directory_compile}/model.rocm_{rocm_chip}.vmfb",
            f"--parameters=model={directory}/{self.real_weights_file_name}",
            f"--function={self.function_run}",
            f"--benchmark_repetitions={self.benchmark_repetitions}",
            f"--benchmark_min_warmup_time={self.benchmark_min_warmup_time}",
            "--benchmark_format=json"
        ] + get_input_list(self.inputs)
        
        # If there are modules for an e2e pipeline test, reset exec_args and directory_compile variables to custom variables
        if self.modules:
            exec_args = e2e_iree_benchmark_module_args(self.modules, self.compiled_file_name)
            exec_args += [
                f"--function={self.function_run}",
                f"--benchmark_repetitions={self.benchmark_repetitions}",
                f"--benchmark_min_warmup_time={self.benchmark_min_warmup_time}"
            ] + get_input_list(self.inputs)
            
            directory_compile = f"{vmfb_dir}/{self.compiled_file_name}_rocm.vmfb"

        # run iree benchmark command
        ret_value, output = run_iree_command(exec_args)    
        # parse the output and retrieve the benchmark mean time
        benchmark_mean_time = job_summary_process(ret_value, output, self.model_name)

        """
        Golden value checks
        - Check all values are either <= than golden values for times and == for compilation statistics.
        """
        # golden time check
        if self.golden_time:
            logger.info((
                f"{self.model_name} {self.submodel_name} benchmark time: {str(benchmark_mean_time)} ms"
                f" (golden time {self.golden_time} ms)"
            ))

            check.less_equal(
                benchmark_mean_time, 
                self.golden_time * self.golden_time_tolerance_multiplier, 
                f"{self.model_name} {self.submodel_name} benchmark time should not regress more than a factor of {self.golden_time_tolerance_multiplier}"
            )
        
        # golden dispatch check
        if self.golden_dispatch:
            with open(f"{directory_compile}/compilation_info.json", "r") as file:
                comp_stats = json.load(file)
            dispatch_count = int(comp_stats["stream-aggregate"]["execution"]["dispatch-count"])
            logger.info((
                f"{self.model_name} {self.submodel_name} dispatch count: {dispatch_count}"
                f" (golden dispatch count {self.golden_dispatch})"
            ))
            check.less_equal(
                dispatch_count,
                self.golden_dispatch,
                f"{self.model_name} {self.submodel_name} dispatch count should not regress"
            )
        
        # golden size check
        if self.golden_size:
            module_path = f"{directory_compile}/model.rocm_{rocm_chip}.vmfb"
            binary_size = Path(module_path).stat().st_size
            logger.info((
                f"{self.model_name} {self.submodel_name} binary size: {binary_size} bytes"
                f" (golden binary size {self.golden_size} bytes)"
            ))

            check.less_equal(
                binary_size,
                self.golden_size,
                f"{self.model_name} {self.submodel_name} binary size should not get bigger"
            )
