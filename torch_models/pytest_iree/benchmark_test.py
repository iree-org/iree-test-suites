# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import logging
from pathlib import Path
import logging

from pytest_iree.test_base import TestBase
from pytest_iree.utils import iree_benchmark_module

logger = logging.getLogger(__name__)


class IREEBenchmarkTest(TestBase):
    """
    A test case for benchmarking IREE modules for given input data.
    """

    def __init__(self, *, test_data: dict, temp_working_dir: Path, **kwargs):
        super().__init__(
            test_data=test_data, temp_working_dir=temp_working_dir, **kwargs
        )
        self.add_marker("benchmark")
        self.golden_time = test_data.get("golden_time_ms", None)
        self.module_artifacts = self._get_modules()
        self.weight_artifacts = self._get_weights()
        self.mean_time = None

    def _get_mean_time_from_output_json(self, output_json: dict) -> float:
        benchmarks = output_json["benchmarks"]
        for benchmark in benchmarks:
            if "aggregate_name" in benchmark and benchmark["aggregate_name"] == "mean":
                return float(benchmark["real_time"])
        # Fallback to the first benchmark's real_time if no mean aggregate is found.
        return benchmarks[0]["real_time"]

    def runtest(self):
        # TODO: Figure out how to do this with pytest_runtest_makereport instead.
        # Earlier attempts didn't work because pytest would never go into the
        # user defined hook for non python tests.
        try:
            # Compile all required modules.
            for module in self.module_artifacts:
                module.join()
            # Get common run arguments.
            run_args = self._get_common_run_args(self.weight_artifacts)
            # Run the model.
            output_json = iree_benchmark_module(
                modules=[m.path for m in self.module_artifacts],
                cwd=self.artifact_dir,
                args=run_args,
            )
            self.mean_time = self._get_mean_time_from_output_json(output_json)
            if self.golden_time is not None:
                assert (
                    self.mean_time <= self.golden_time
                ), f"Benchmark failed: mean_time {self.mean_time} exceeds golden_time {self.golden_time}"
            self.status = "PASSED"
        except Exception as e:
            self.status = "FAILED"
            raise e

    @classmethod
    def get_test_type(cls) -> str:
        return "benchmark"

    @classmethod
    def get_test_headers(cls) -> list[str]:
        return ["Name", "Current Time (ms)", "Golden Time (ms)", "Status"]

    def get_test_summary(self) -> list:
        return [
            self.name,
            f"{self.mean_time:.3f}" if self.mean_time is not None else "N/A",
            f"{self.golden_time:.3f}" if self.golden_time is not None else "N/A",
            self.status,
        ]
