# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import glob
import json
import numpy as np
import os
from pathlib import Path
import pytest
import subprocess

pytest.register_assert_rewrite('generate_tests')
from generate_tests import GenConfig, IreeCompileException, IreeRunException, IreeXFailCompileRunException

THIS_DIR = Path(__file__).parent
TEST_DATA_FLAGFILE_NAME = "run_module_io.json"


def pytest_addoption(parser):
    # List of configuration files following this schema:
    #   {
    #     "config_name": str,
    #     "iree_compile_flags": list of str,
    #     "iree_run_module_flags": list of str,
    #     "skip_compile_tests": list of str,
    #     "skip_run_tests": list of str,
    #     "expected_compile_failures": list of str,
    #     "expected_run_failures": list of str
    #   }
    #
    # For example, to test on CPU with the `llvm-cpu` backend and `local-task` device:
    #   {
    #     "config_name": "cpu_llvm_task",
    #     "iree_compile_flags": ["--iree-hal-target-backends=llvm-cpu"],
    #     "iree_run_module_flags": ["--device=local-task"],
    #     "skip_compile_tests": [],
    #     "skip_run_tests": [],
    #     "expected_compile_failures": ["test_abs"],
    #     "expected_run_failures": ["test_add"],
    #   }
    #
    # The list of files can be specified in (by order of preference):
    #   1. The `--config-files` argument
    #       e.g. `pytest ... --config-files foo.json bar.json`
    #   2. The `IREE_TEST_CONFIG_FILES` environment variable
    #       e.g. `set IREE_TEST_CONFIG_FILES=foo.json;bar.json`
    #   3. A default config file used for testing the test suite itself
    default_config_files = [
        f for f in os.getenv("IREE_TEST_CONFIG_FILES", "").split(";") if f
    ]
    if not default_config_files:
        default_config_files = [
            THIS_DIR / "configs" / "torch_ops_cpu_llvm_sync.json",
        ]
    parser.addoption(
        "--config-files",
        action="store",
        nargs="*",
        default=default_config_files,
        help="List of config JSON files used to build test cases",
    )

    parser.addoption(
        "--ignore-xfails",
        action="store_true",
        default=False,
        help="Ignores expected compile/run failures from configs, to print all error output",
    )

    parser.addoption(
        "--skip-all-runs",
        action="store_true",
        default=False,
        help="Skips all 'run' tests, overriding 'skip_run_tests' in configs",
    )


def pytest_sessionstart(session):
    session.config.iree_test_configs = []
    for config_file in session.config.getoption("config_files"):
        with open(config_file) as f:
            test_config = json.load(f)

            # Sanity check the config file structure before going any further.
            def check_field(field_name):
                if field_name not in test_config:
                    raise ValueError(
                        f"config file '{config_file}' is missing a '{field_name}' field"
                    )

            check_field("config_name")
            check_field("iree_compile_flags")
            check_field("iree_run_module_flags")

            session.config.iree_test_configs.append(test_config)


def pytest_collect_file(parent, file_path):
    """A test corresponds to a file named ${TEST_DATA_FLAGFILE_NAME}."""
    if file_path.name == TEST_DATA_FLAGFILE_NAME:
        return MlirCompileRunTest.from_parent(parent, path=file_path)

class MlirCompileRunTest(pytest.File):
    """Collector for MLIR -> compile -> run tests anchored on a run_module_io.json file."""

    @property
    def ignore_xfails(self):
        return self.config.getoption("ignore_xfails")

    @property
    def skip_all_runs(self):
        return self.config.getoption("skip_all_runs")

    def expect_compile_success(self, tgt_config, gen_config):
        expected_comp_fails = tgt_config.get("expected_compile_failures", [])
        return self.ignore_xfails or gen_config.qualified_name not in expected_comp_fails

    def expect_run_success(self, tgt_config, gen_config):
        expected_run_fails = tgt_config.get("expected_run_failures", [])
        return self.ignore_xfails or gen_config.qualified_name not in expected_run_fails

    def skip_run(self, tgt_config, gen_config):
        skips = tgt_config.get("skip_run_tests", [])
        return self.skip_all_runs or gen_config.qualified_name in skips

    def collect(self):
        # self.path is run_module_io.json
        gen_config = GenConfig.load(self.path)

        for tgt_config in self.config.iree_test_configs:
            if gen_config.qualified_name in tgt_config.get("skip_compile_tests", []):
                continue

            tgt_config["expect_compile_success"] = self.expect_compile_success(tgt_config, gen_config)
            tgt_config["expect_run_success"] = self.expect_run_success(tgt_config, gen_config)
            tgt_config["skip_run"] = self.skip_run(tgt_config, gen_config)
            tgt_config["golden_time_ms"] = tgt_config.get("golden_times_ms", {}).get(gen_config.qualified_name, float("nan"))

            match gen_config.mode:
                case "compare":
                    cls = IreeCompareTest
                case "benchmark":
                    cls = IreeBenchmarkTest

            yield cls.from_parent(self, name=gen_config.qualified_name, tgt_config=copy.copy(tgt_config), gen_config=gen_config)

class IreeBaseTest(pytest.Item):
    def __init__(self, tgt_config, gen_config, **kwargs):
        super().__init__(**kwargs)
        print(2, gen_config.qualified_name, tgt_config["golden_time_ms"])
        self.tgt_config = tgt_config
        print(3, gen_config.qualified_name, self.tgt_config["golden_time_ms"])
        self.gen_config = gen_config
        self.add_markers()

    @property
    def iree_compile_flags(self):
        return self.tgt_config["iree_compile_flags"]

    @property
    def iree_run_flags(self):
        return self.tgt_config["iree_run_module_flags"]

    @property
    def skip_run(self):
        return self.tgt_config["skip_run"]

    @property
    def expect_compile_success(self):
        return self.tgt_config["expect_compile_success"]

    @property
    def qualified_name(self):
        return self.gen_config.qualified_name

    @property
    def golden_time(self):
        print(3, self.gen_config.qualified_name, self.tgt_config["golden_time_ms"])
        return self.tgt_config["golden_time_ms"]

    def repr_failure(self, excinfo):
        """Called when self.runtest() raises an exception."""
        if isinstance(excinfo.value, (IreeCompileException, IreeRunException)):
            return "\n".join(excinfo.value.args)
        if isinstance(excinfo.value, IreeXFailCompileRunException):
            return (
                "Expected compile failure but run failed (move to 'expected_run_failures'):\n"
                + "\n".join(excinfo.value.__cause__.args)
            )
        return super().repr_failure(excinfo)


    def add_markers(self):
        if not self.tgt_config["expect_compile_success"]:
            self.add_marker(
                pytest.mark.xfail(
                    raises=IreeCompileException,
                    strict=True,
                    reason="Expected compilation to fail (included in 'expected_compile_failures')",
                )
            )
        if not self.tgt_config["expect_run_success"]:
            self.add_marker(
                pytest.mark.xfail(
                    raises=IreeRunException,
                    strict=True,
                    reason="Expected run to fail (included in 'expected_run_failures')",
                )
            )

class IreeCompareTest(IreeBaseTest):
    def runtest(self):
        iree_compile_flags = self.iree_compile_flags
        iree_run_flags = self.iree_run_flags
        skip_run = self.skip_run
        try:
            self.gen_config.run_quality_test(iree_compile_flags, iree_run_flags, skip_run)
        except IreeRunException as e:
            if not self.expect_compile_success:
                raise IreeXFailCompileRunException from e
            raise e

    def reportinfo(self):
        return self.path, 0, f"IREE quality test: {self.qualified_name}"

class IreeBenchmarkTest(IreeBaseTest):
    def runtest(self):
        iree_compile_flags = self.iree_compile_flags
        iree_run_flags = self.iree_run_flags
        skip_run = self.skip_run
        try:
            self.gen_config.run_benchmark_test(iree_compile_flags, iree_run_flags, golden_time=self.golden_time, skip_run=skip_run)
        except IreeRunException as e:
            if not self.expect_compile_success:
                raise IreeXFailCompileRunException from e
            raise e

    def reportinfo(self):
        return self.path, 0, f"IREE benchmark test: {self.qualified_name}"
