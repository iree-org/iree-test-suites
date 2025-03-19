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
backend = os.getenv("BACKEND", default="rocm")

def pytest_addoption(parser):
    parser.addoption(
        "--test-file-directory",
        action="store",
        help="The directory of quality test JSON files to build and run test cases",
    )

    parser.addoption(
        "--external-file-directory",
        action="store",
        help="The directory of external test files (ex: E2E MLIR, tuner files)",
    )


def pytest_configure():
    pytest.vmfb_manager = {}


def pytest_sessionstart(session):
    logger.info("Pytest quality test session is starting")

    # Collect all .json files for quality tests
    session.config.quality_test_files = []
    path_of_quality_tests = Path(session.config.getoption("test_file_directory"))
    test_files = sorted(path_of_quality_tests.glob("**/*.json"))
    for test_file in test_files:
        if backend in str(test_file.name):
            session.config.quality_test_files.append(test_file)

    # Keeping track of all external test files and their paths
    session.config.external_test_files = {}
    path_of_external_test_files = Path(
        session.config.getoption("external_file_directory")
    )
    external_files = sorted(path_of_external_test_files.glob("*"))
    for external_file in external_files:
        file_name = external_file.name
        session.config.external_test_files[file_name] = external_file


def pytest_collect_file(parent, file_path):
    # Run only the quality test for this directory
    if "model_quality_run" in str(file_path):
        return SharkTankModelQualityTests.from_parent(parent, path=file_path)

@dataclass(frozen=True)
class QualityTestSpec:
    model_name: str
    quality_file_name: str
    file_path: Path
    external_test_files: dict


class SharkTankModelQualityTests(pytest.File):
    def collect(self):
        for file_path in self.config.quality_test_files:
            quality_file_name = file_path.stem
            model_name = str(file_path.parent.stem)

            item_name = f"{model_name} :: {quality_file_name}"

            spec = QualityTestSpec(
                model_name=model_name,
                quality_file_name=quality_file_name,
                file_path=file_path,
                external_test_files=self.config.external_test_files,
            )

            yield ModelQualityRunItem.from_parent(self, name=item_name, spec=spec)
