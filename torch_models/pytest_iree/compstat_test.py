# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import logging
from pathlib import Path
import logging

from pytest_iree.utils import iree_run_module
from pytest_iree.test_base import TestBase
from pytest_iree.module import ModuleArtifact

logger = logging.getLogger(__name__)


class IREECompStatTest(TestBase):
    def __init__(self, *, test_data: dict, **kwargs):
        super().__init__(test_data=test_data, **kwargs)
        self.add_marker("compstat")
        module = test_data.get("module", None)
        assert module is not None, "Test data must contain 'module' field"
        self.module_artifact = ModuleArtifact(
            artifact_base_dir=self.artifact_dir,
            module_base_dir=self.module_directory,
            module=module,
            external_file_dir=self.external_file_directory,
            force_recompile=self.force_recompile,
        )
        self.golden_dispatch_count = test_data.get("golden_dispatch_count", None)
        self.golden_binary_size = test_data.get("golden_binary_size", None)
        self.current_dispatch_count = None
        self.current_binary_size = None

    def runtest(self):
        # No matter what I do, the pytest_runtest_makereport hook doesn't work.
        # So workaround by setting status directly here.
        try:
            # Get compilation stats.
            compstats = self.module_artifact.get_compstats()
            self.current_dispatch_count = (
                compstats.get("stream-aggregate", {})
                .get("execution", {})
                .get("dispatch-count", None)
            )
            self.current_binary_size = self.module_artifact.path.stat().st_size
            if self.golden_dispatch_count is not None:
                assert (
                    self.current_dispatch_count is not None
                ), "Current dispatch count is None"
                assert (
                    self.current_dispatch_count <= self.golden_dispatch_count
                ), f"Dispatch count mismatch: expected {self.golden_dispatch_count}, got {self.current_dispatch_count}"
            self.status = "PASSED"
            if self.golden_binary_size is not None:
                assert (
                    self.current_binary_size is not None
                ), "Current binary size is None"
                assert (
                    self.current_binary_size <= self.golden_binary_size
                ), f"Binary size mismatch: expected {self.golden_binary_size}, got {self.current_binary_size}"
        except Exception as e:
            self.status = "FAILED"
            raise e

    @classmethod
    def get_test_type(cls) -> str:
        return "compstat"

    @classmethod
    def get_test_headers(cls) -> list[str]:
        return [
            "Name",
            "Golden Dispatch Count",
            "Current Dispatch Count",
            "Golden Binary Size (bytes)",
            "Current Binary Size (bytes)",
            "Status",
        ]

    def get_test_summary(self) -> list:
        return [
            self.name,
            self.golden_dispatch_count
            if self.golden_dispatch_count is not None
            else "N/A",
            self.current_dispatch_count
            if self.current_dispatch_count is not None
            else "N/A",
            self.golden_binary_size if self.golden_binary_size is not None else "N/A",
            self.current_binary_size if self.current_binary_size is not None else "N/A",
            self.status,
        ]
