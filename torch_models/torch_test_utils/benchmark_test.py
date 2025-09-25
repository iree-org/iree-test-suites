# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import logging
from pathlib import Path
import logging

from torch_test_utils.test_base import TestBase
from torch_test_utils.utils import iree_benchmark_module

logger = logging.getLogger(__name__)


class TorchModelBenchmarkTest(TestBase):
    def __init__(self, *, test_data: dict, **kwargs):
        super().__init__(test_data=test_data, **kwargs)
        self.add_marker("benchmark")
        self.golden_time = test_data.get("golden_time", None)
        self.module_artifacts = self._get_modules()

    def _get_mean_time_from_output_json(self, output_json: dict) -> float:
        benchmarks = output_json["benchmarks"]
        for benchmark in benchmarks:
            if "aggregate_name" in benchmark and benchmark["aggregate_name"] == "mean":
                return float(benchmark["real_time"])
        # Fallback to the first benchmark's real_time if no mean aggregate is found.
        return benchmarks[0]["real_time"]

    def runtest(self):
        # Compile all required modules.
        for module in self.module_artifacts:
            module.join()
        # Get common run arguments.
        run_args = self._get_common_run_args()
        # Run the model.
        output_json = iree_benchmark_module(
            modules=[m.path for m in self.module_artifacts],
            cwd=self.artifact_dir,
            args=run_args,
        )
        mean_time = self._get_mean_time_from_output_json(output_json)
        if self.golden_time is not None:
            assert (
                mean_time <= self.golden_time
            ), f"Benchmark failed: mean_time {mean_time} exceeds golden_time {self.golden_time}"
