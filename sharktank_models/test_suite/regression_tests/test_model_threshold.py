# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers_tools import *
import os
from conftest import VmfbManager
from pathlib import Path
import subprocess
import json 

rocm_chip = os.getenv("ROCM_CHIP", default="gfx942")
vmfb_dir = os.getenv("TEST_OUTPUT_ARTIFACTS", default=Path.cwd()) 
sku = os.getenv("SKU", default="mi300")

# Helper methods
def fetch_source_fixtures_for_run_flags(inference_list, model_name, submodel_name):
    result = []
    for entry in inference_list:
        source = entry.get("source")
        value = entry.get("value")
        source_fixture = fetch_source_fixture(source, group=f"{model_name}_{submodel_name}")
        result.append([source_fixture.path, value])
    
    return result
    
def common_run_flags_generation(input_list, output_list):
    flags_list = []

    if input_list:
        for path, value in input_list:
            if not value:
                flags_list.append(f"--input=@{path}")
            else:
                flags_list.append(f"--input={value}=@{path}")

    if output_list:
        for path, value in output_list:
            if not value:
                flags_list.append(f"--expected_output=@{path}")
            else:
                flags_list.append(f"--expected_output={value}=@{path}")
    
    return flags_list


class TestModelThreshold:
    @pytest.fixture(autouse = True)
    @classmethod
    def setup_class(self, pytestconfig):
        self.model_name = pytestconfig.getoption("model_name")
        self.submodel_name = pytestconfig.getoption("submodel_name")

        file_name = f"{Path.cwd()}/sharktank_models/test_suite/regression_tests/{self.model_name}/{self.submodel_name}.json"

        with open(file_name ,'r') as file:
            data = json.load(file)

            # retrieving source fixtures if available in JSON file
            self.inputs = fetch_source_fixtures_for_run_flags(data.get("inputs"), self.model_name, self.submodel_name) if data.get("inputs") else None
            self.outputs = fetch_source_fixtures_for_run_flags(data.get("outputs"), self.model_name, self.submodel_name) if data.get("outputs") else None
            self.real_weights = fetch_source_fixture(data.get("real_weights"), group=f"{self.model_name}_{self.submodel_name}") if data.get("real_weights") else None
            self.mlir = fetch_source_fixture(data.get("mlir"), group=f"{self.model_name}_{self.submodel_name}") if data.get("mlir") else None

            # setting compiler options for cpu and rocm
            self.cpu_compiler_flags = data.get("cpu_compiler_flags", [])
            self.cpu_compiler_flags.append("--iree-hal-target-backends=llvm-cpu")

            self.rocm_compiler_flags = data.get("rocm_compiler_flags", [])
            self.rocm_compiler_flags.append("--iree-hal-target-backends=rocm")
            self.rocm_compiler_flags.append(f"--iree-hip-target={rocm_chip}")

            # Setting input, output, and function call arguments
            self.common_rule_flags = common_run_flags_generation(self.inputs, self.outputs)
            self.cpu_threshold_args = data.get("cpu_threshold_args", [])
            self.rocm_threshold_args = data.get("rocm_threshold_args", [])
            self.run_cpu_function = data.get("run_cpu_function")
            self.run_rocm_function = data.get("run_rocm_function")

            # Custom configurations for selecting tests to fail or ignoring certain tests
            self.compile_only = data.get("compile_only", False)
            self.cpu_run_test_expecting_to_fail = data.get("cpu_run_test_expecting_to_fail", False)
            self.rocm_run_test_expecting_to_fail = data.get("rocm_run_test_expecting_to_fail", False)
            self.rocm_compile_chip_expecting_to_fail = data.get("rocm_compile_chip_expecting_to_fail", [])
            self.rocm_tests_only = data.get("rocm_tests_only", False)

            # Custom configuration for a tuner file
            self.tuner_file = data.get("tuner_file", {})
            if sku in self.tuner_file:
                self.rocm_compiler_flags.append(f"--iree-codegen-transform-dialect-library={Path.cwd()}/{self.tuner_file.get(sku)}")

            # Custom configuration to fp16 and adding secondary pipeline mlir
            self.rocm_pipeline_compiler_flags = data.get("rocm_pipeline_compiler_flags", [])
            self.rocm_pipeline_compiler_flags.append("--iree-hal-target-backends=rocm")
            self.rocm_pipeline_compiler_flags.append(f"--iree-hip-target={rocm_chip}")
            self.pipeline_mlir = fetch_source_fixture(data.get("pipeline_mlir"), group=f"{self.model_name}_{self.submodel_name}") if data.get("pipeline_mlir") else None
            self.add_pipeline_module = data.get("add_pipeline_module", False)
            

    ###############################################################################
    # CPU
    ###############################################################################
    def test_compile_cpu(self):
        if self.rocm_tests_only:
            pytest.skip("Only ROCM tests are being run, skipping CPU tests...")

        vmfbs_path = f"{self.model_name}_{self.submodel_name}_vmfbs"
        VmfbManager.cpu_vmfb = iree_compile(
            self.mlir,
            self.cpu_compiler_flags,
            Path(vmfb_dir)
            / Path(vmfbs_path)
            / Path('model').with_suffix(f".cpu.vmfb"),
        )

        if self.pipeline_mlir:
            VmfbManager.pipeline_cpu_vmfb = iree_compile(
                self.pipeline_mlir,
                self.cpu_compiler_flags,
                Path(vmfb_dir)
                / Path(vmfbs_path)
                / Path('pipeline_model').with_suffix(f".cpu.vmfb"),
            )

    @pytest.mark.depends(on=["test_compile_cpu"])
    def test_run_cpu_threshold(self):
        if self.compile_only:
            pytest.skip("Only compilation tests are selected, skipping threshold test...")
        
        if self.cpu_run_test_expecting_to_fail:
            pytest.xfail("Expected run to fail")

        args = self.cpu_threshold_args + self.common_rule_flags
        if self.real_weights:
            args.append(f"--parameters=model={self.real_weights.path}")

        if self.add_pipeline_module:
            args.append(f"--module={VmfbManager.pipeline_cpu_vmfb}")

        iree_run_module(
            VmfbManager.cpu_vmfb,
            device="local-task",
            function=self.run_cpu_function,
            args=args
        )

    ###############################################################################
    # ROCM
    ###############################################################################

    def test_compile_rocm(self):
        if rocm_chip in self.rocm_compile_chip_expecting_to_fail:
            pytest.xfail(f"Expecting {rocm_chip} compilation to fail for {self.submodel_name}")

        vmfbs_path = f"{self.model_name}_{self.submodel_name}_vmfbs"
        VmfbManager.rocm_vmfb = iree_compile(
            self.mlir,
            self.rocm_compiler_flags,
            Path(vmfb_dir)
            / Path(vmfbs_path)
            / Path('model').with_suffix(f".rocm_{rocm_chip}.vmfb"),
        )

        if self.pipeline_mlir:
            VmfbManager.pipeline_rocm_vmfb = iree_compile(
                self.pipeline_mlir,
                self.rocm_pipeline_compiler_flags,
                Path(vmfb_dir)
                / Path(vmfbs_path)
                / Path('pipeline_model').with_suffix(f".rocm_{rocm_chip}.vmfb"),
            )

    @pytest.mark.depends(on=["test_compile_rocm"])
    def test_run_rocm_threshold(self):
        if self.compile_only:
            pytest.skip("Only compilation tests are selected, skipping threshold test...")
        
        if self.rocm_run_test_expecting_to_fail:
            pytest.xfail("Expected run to fail")

        args = self.rocm_threshold_args + self.common_rule_flags
        if self.real_weights:
            args.append(f"--parameters=model={self.real_weights.path}")

        if self.add_pipeline_module:
            args.append(f"--module={VmfbManager.pipeline_rocm_vmfb}")

        return iree_run_module(
            VmfbManager.rocm_vmfb,
            device="hip",
            function=self.run_rocm_function,
            args=args
        )