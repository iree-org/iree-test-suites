# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers_tools import *
import os
from pathlib import Path
import subprocess
import json

THIS_DIR = Path(__file__).parent
# compiled files will live in the previous directory, so benchmark tests can access those and no need to recompile
PARENT_DIR = Path(__file__).parent.parent
vmfb_dir = os.getenv("TEST_OUTPUT_ARTIFACTS", default=str(PARENT_DIR))
chip = os.getenv("ROCM_CHIP", default="gfx942")
sku = os.getenv("SKU", default="mi300")

# Helper methods
def fetch_source_fixtures_for_run_flags(inference_list, model_name, submodel_name):
    result = []
    for entry in inference_list:
        source = entry.get("source")
        value = entry.get("value")
        source_fixture = fetch_source_fixture(
            source, group=f"{model_name}_{submodel_name}"
        )
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


class ModelQualityRunItem(pytest.Item):
    
    def __init__(self, spec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self.model_name = self.spec.model_name
        self.quality_file_name = self.spec.quality_file_name
        SUBMODEL_FILE_PATH = THIS_DIR / f"{self.model_name}/{self.quality_file_name}.json"
        split_file_name = self.quality_file_name.split("_")
        self.submodel_name = "_".join(split_file_name[:-1])
        self.type_of_backend = split_file_name[-1]

        with open(SUBMODEL_FILE_PATH, "r") as file:
            data = json.load(file)

            # retrieving source fixtures if available in JSON file
            self.inputs = (
                fetch_source_fixtures_for_run_flags(
                    data.get("inputs"), self.model_name, self.submodel_name
                )
                if data.get("inputs")
                else None
            )
            self.outputs = (
                fetch_source_fixtures_for_run_flags(
                    data.get("outputs"), self.model_name, self.submodel_name
                )
                if data.get("outputs")
                else None
            )
            self.real_weights = (
                fetch_source_fixture(
                    data.get("real_weights"),
                    group=f"{self.model_name}_{self.submodel_name}",
                )
                if data.get("real_weights")
                else None
            )
            self.mlir = (
                fetch_source_fixture(
                    data.get("mlir"), group=f"{self.model_name}_{self.submodel_name}"
                )
                if data.get("mlir")
                else None
            )
            
            self.compiler_flags = data.get("compiler_flags", [])            
            self.device = data.get("device")

            # Setting input, output, and function call arguments
            self.common_rule_flags = common_run_flags_generation(
                self.inputs, self.outputs
            )
            self.threshold_args = data.get("threshold_args", [])
            self.run_function = data.get("run_function")

            # Custom configurations for selecting tests to fail or ignoring certain tests
            self.compile_only = data.get("compile_only", False)
            self.run_test_expecting_to_fail = data.get(
                "run_test_expecting_to_fail", False
            )
            self.compile_chip_expecting_to_fail = data.get(
                "compile_chip_expecting_to_fail", []
            )

            # Custom configuration for a tuner file
            self.tuner_file = data.get("tuner_file", {})
            if sku in self.tuner_file:
                TUNER_FILE_PATH = THIS_DIR / self.tuner_file.get(sku)
                self.compiler_flags.append(
                    f"--iree-codegen-transform-dialect-library={str(TUNER_FILE_PATH)}"
                )

            # Custom configuration to fp16 and adding secondary pipeline mlir
            self.pipeline_compiler_flags = data.get(
                "pipeline_compiler_flags", []
            )
            self.pipeline_mlir = (
                fetch_source_fixture(
                    data.get("pipeline_mlir"),
                    group=f"{self.model_name}_{self.submodel_name}",
                )
                if data.get("pipeline_mlir")
                else None
            )
            self.add_pipeline_module = data.get("add_pipeline_module", False)
            
            if self.type_of_backend == "rocm":
                self.file_suffix = f"{self.type_of_backend}_{chip}"
                self.compiler_flags += [
                    f"--iree-hip-target={chip}"
                ]
                self.pipeline_compiler_flags.append(f"--iree-hip-target={chip}")

            elif self.type_of_backend == "cpu":
                self.file_suffix = "cpu"
            
    def runtest(self):
        self.test_compile()
        self.test_run_threshold()


    def test_compile(self):
        if chip in self.compile_chip_expecting_to_fail:
            pytest.xfail(
                f"Expecting {chip} compilation to fail for {self.submodel_name}"
            )

        vmfbs_path = f"{self.model_name}_{self.submodel_name}_vmfbs"
        vmfb_manager_unique_key = f"{self.model_name}_{self.submodel_name}_{self.type_of_backend}_vmfb"
        pytest.vmfb_manager[vmfb_manager_unique_key] = iree_compile(
            self.mlir,
            self.compiler_flags,
            Path(vmfb_dir)
            / Path(vmfbs_path)
            / Path("model").with_suffix(f".{self.type_of_backend}_{chip}.vmfb"),
        )

        if self.pipeline_mlir:
            pipeline_vmfb_manager_unique_key = (
                f"{self.model_name}_{self.submodel_name}_pipeline_{self.type_of_backend}_vmfb"
            )
            pytest.vmfb_manager[pipeline_vmfb_manager_unique_key] = iree_compile(
                self.pipeline_mlir,
                self.pipeline_compiler_flags,
                Path(vmfb_dir)
                / Path(vmfbs_path)
                / Path("pipeline_model").with_suffix(f".{self.type_of_backend}_{chip}.vmfb"),
            )

    def test_run_threshold(self):
        if self.compile_only:
            pytest.skip(
                "Only compilation tests are selected, skipping threshold test..."
            )

        if self.run_test_expecting_to_fail:
            pytest.xfail("Expected run to fail")

        args = self.threshold_args + self.common_rule_flags
        if self.real_weights:
            args.append(f"--parameters=model={self.real_weights.path}")

        if self.add_pipeline_module:
            pipeline_vmfb_manager_unique_key = (
                f"{self.model_name}_{self.submodel_name}_pipeline_{self.type_of_backend}_vmfb"
            )
            pipeline_module_name = pytest.vmfb_manager.get(
                pipeline_vmfb_manager_unique_key
            )
            args.append(f"--module={pipeline_module_name}")

        vmfb_manager_unique_key = f"{self.model_name}_{self.submodel_name}_{self.type_of_backend}_vmfb"
        return iree_run_module(
            Path(pytest.vmfb_manager.get(vmfb_manager_unique_key)),
            device=self.device,
            function=self.run_function,
            args=args,
        )

    def repr_failure(self, excinfo):
        return super().repr_failure(excinfo)
        
    def reportinfo(self):
        return self.path, 0, f"usecase: {self.name}"
