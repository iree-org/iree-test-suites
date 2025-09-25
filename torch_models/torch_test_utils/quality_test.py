# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import logging
from pathlib import Path
import logging

from torch_test_utils.utils import iree_run_module
from torch_test_utils.test_base import TestBase

logger = logging.getLogger(__name__)


class TorchModelQualityTest(TestBase):
    def __init__(self, *, test_data: dict, **kwargs):
        super().__init__(test_data=test_data, **kwargs)
        self.add_marker("quality")
        self.module_artifacts = self._get_modules()

    def runtest(self):
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
