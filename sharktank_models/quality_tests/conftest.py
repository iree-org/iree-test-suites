# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from model_quality_run import ModelQualityRunItem
from pathlib import Path
import os
import logging
from dataclasses import dataclass

THIS_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

def pytest_configure():
    pytest.vmfb_manager = {}

def pytest_sessionstart(session):
    logger.info("Pytest quality test session is starting")

def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".json" and "quality_tests" in str(THIS_DIR):
        return SharkTankModelQualityTests.from_parent(parent, path=file_path)

@dataclass(frozen = True)
class QualityTestSpec:
    model_name: str
    quality_file_name: str
    
class SharkTankModelQualityTests(pytest.File):
    
    def collect(self):
        path = str(self.path).split("/")
        quality_file_name = path[-1].replace(".json", "")
        model_name = path[-2]
        
        item_name = f"{model_name} :: {quality_file_name}"
        
        spec = QualityTestSpec(
            model_name = model_name,
            quality_file_name = quality_file_name
        )
        
        yield ModelQualityRunItem.from_parent(self, name=item_name, spec=spec)
    