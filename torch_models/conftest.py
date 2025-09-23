# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import logging
from pathlib import Path
import logging
import json

from torch_test_utils.quality_test import TorchModelQualityTest
from torch_test_utils.benchmark_test import TorchModelBenchmarkTest

logger = logging.getLogger(__name__)


@pytest.hookimpl
def pytest_addoption(parser):
    # Required options.
    parser.addoption(
        "--test-file-directory",
        action="store",
        help="The directory of test JSON test cases",
    )
    parser.addoption(
        "--module-directory",
        action="store",
        default=None,
        help="The directory of torch models to be compiled",
    )
    parser.addoption(
        "--external-file-directory",
        action="store",
        default=None,
        help="The directory of external test files (ex: E2E MLIR, tuner files)",
    )

    # Optional options.
    parser.addoption(
        "--artifact-directory",
        action="store",
        default="artifacts",
        help="The base directory to store compiled/downloaded artifacts.",
    )
    parser.addoption(
        "--force-recompile",
        action="store_true",
        default=False,
        help="Force recompilation of modules even if cached versions exist.",
    )


@pytest.hookimpl
def pytest_sessionstart(session):
    # Check if all required options are provided.
    required_options = [
        "--test-file-directory",
        "--module-directory",
        "--external-file-directory",
    ]
    for option in required_options:
        if session.config.getoption(option) is None:
            raise ValueError(f"{option} is required but not provided.")


class TorchModelDirectory(pytest.Directory):
    def collect(self):
        test_files: list[Path] = sorted(self.path.glob("**/*.json"))
        for test_file in test_files:
            yield TorchModelFile.from_parent(path=test_file, parent=self)


class TorchModelFile(pytest.File):
    def collect(self):
        test_data = json.loads(self.path.read_text())
        if test_data["type"] == "quality":
            yield TorchModelQualityTest.from_parent(
                self, test_data=test_data, name=self.name
            )
        elif test_data["type"] == "benchmark":
            yield TorchModelBenchmarkTest.from_parent(
                self, test_data=test_data, name=self.name
            )
        else:
            raise ValueError(f"Unknown test type: {test_data['type']}")


@pytest.hookimpl
def pytest_collect_file(file_path: Path, parent: pytest.Collector):
    if file_path == Path(__file__):
        test_file_directory = parent.config.getoption("test_file_directory")
        assert (
            test_file_directory is not None
        ), "--test-file-directory must be specified"
        test_file_directory = Path(test_file_directory).resolve()
        return TorchModelDirectory.from_parent(parent=parent, path=test_file_directory)
