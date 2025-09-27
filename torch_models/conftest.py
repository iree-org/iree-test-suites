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
import tabulate

from pytest_iree.test_base import TestBase
from pytest_iree.quality_test import IREEQualityTest
from pytest_iree.benchmark_test import IREEBenchmarkTest
from pytest_iree.compstat_test import IREECompStatTest

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
        help="The directory of iree models to be compiled",
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
    parser.addoption(
        "--job-summary-path",
        action="store",
        default="./",
        help="The directory to store the job summary markdown file.",
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


class IREEDirectory(pytest.Directory):
    def collect(self):
        test_files: list[Path] = sorted(self.path.glob("**/*.json"))
        for test_file in test_files:
            yield IREEFile.from_parent(path=test_file, parent=self)


class IREEFile(pytest.File):
    def collect(self):
        test_data = json.loads(self.path.read_text())
        if "type" not in test_data:
            # Not a valid JSON test file.
            return
        if test_data["type"] == "quality":
            yield IREEQualityTest.from_parent(self, test_data=test_data, name=self.name)
        elif test_data["type"] == "benchmark":
            yield IREEBenchmarkTest.from_parent(
                self, test_data=test_data, name=self.name
            )
        elif test_data["type"] == "compstat":
            yield IREECompStatTest.from_parent(
                self, test_data=test_data, name=self.name
            )
        else:
            raise ValueError(f"Unknown test type: {test_data['type']}")


@pytest.hookimpl
def pytest_collect_file(file_path: Path, parent: pytest.Collector):
    if file_path == Path(__file__):
        test_file_directory = parent.config.getoption("test_file_directory")
        logging.info(f"Collecting tests from {test_file_directory}")
        assert (
            test_file_directory is not None
        ), "--test-file-directory must be specified"
        test_file_directory = Path(test_file_directory).resolve()
        return IREEDirectory.from_parent(parent=parent, path=test_file_directory)


@pytest.hookimpl
def pytest_sessionfinish(session, exitstatus):
    # Generate a job_summary report at the end of the session.
    summaries = {}
    for item in session.items:
        assert isinstance(item, TestBase)
        test_type = item.get_test_type()
        if test_type not in summaries:
            summaries[test_type] = {"headers": item.get_test_headers(), "rows": []}
        summaries[test_type]["rows"].append(item.get_test_summary())

    job_summary_path = Path(session.config.getoption("job_summary_path")).resolve()
    with open(job_summary_path / "job_summary.md", "w") as md_file:
        md_file.write(
            f"# Job Summary For Markers: {session.config.getoption('-m')}\n\n"
        )
        for test_type, summary in summaries.items():
            md_file.write(f"## {test_type.capitalize()} Test Summary\n\n")
            table = tabulate.tabulate(
                summary["rows"], headers=summary["headers"], tablefmt="github"
            )
            md_file.write(table + "\n\n")

    with open(job_summary_path / "job_summary.json", "w") as json_file:
        json.dump(summaries, json_file, indent=2)
