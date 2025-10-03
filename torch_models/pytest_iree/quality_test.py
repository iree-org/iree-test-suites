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

logger = logging.getLogger(__name__)


class IREEQualityTest(TestBase):
    """
    A test case for accuracy quality of IREE modules, for given input data.
    """

    def __init__(self, *, test_data: dict, **kwargs):
        super().__init__(test_data=test_data, **kwargs)
        self.add_marker("quality")
        self.module_artifacts = self._get_modules()

    def runtest(self):
        # TODO: Figure out how to do this with pytest_runtest_makereport instead.
        # Earlier attempts didn't work because pytest would never go into the
        # user defined hook for non python tests.
        try:
            # Compile all required modules.
            for module in self.module_artifacts:
                module.join()
            # Get common run arguments.
            run_args = self._get_common_run_args()
            # Run the model.
            iree_run_module(
                modules=[m.path for m in self.module_artifacts],
                cwd=self.artifact_dir,
                args=run_args,
            )
            self.status = "PASSED"
        except Exception as e:
            self.status = "FAILED"
            raise e

    @classmethod
    def get_test_type(cls) -> str:
        return "quality"

    @classmethod
    def get_test_headers(cls) -> list[str]:
        return [
            "Name",
            "Status",
        ]

    def get_test_summary(self) -> list:
        return [
            self.name,
            self.status,
        ]
