# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from model_benchmark_run import ModelBenchmarkRunItem
from pathlib import Path
from dataclasses import dataclass
import logging
import os
import json
import tabulate

THIS_DIR = Path(__file__).parent
sku = os.getenv("SKU", default="mi300")
job_summary_path = os.getenv("JOB_SUMMARY_PATH", str(THIS_DIR))
backend = os.getenv("BACKEND", default="cpu")

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption(
        "--test-file-directory",
        action="store",
        help="The directory of benchmark test JSON files to build and run test cases",
    )

    parser.addoption(
        "--external-file-directory",
        action="store",
        help="The directory of external test files (ex: E2E MLIR, tuner files)",
    )


def pytest_sessionstart(session):
    logger.info("Pytest benchmark test session is starting")
    with open(f"{job_summary_path}/job_summary.md", "a") as job_summary, open(
        f"{job_summary_path}/job_summary.json", "w+"
    ) as content:
        print(f"{sku.upper()} Complete Benchmark Summary:\n", file=job_summary)
        json.dump({}, content)

    # Collect all .json files for benchmark tests
    session.config.benchmark_test_files = []
    path_of_benchmark_tests = Path(session.config.getoption("test_file_directory"))
    model_name = session.config.getoption("-k")
    test_files = sorted(path_of_benchmark_tests.glob("**/*.json"))
    for test_file in test_files:
        if model_name:
            if backend in str(test_file.name) and model_name in str(test_file.name):
                session.config.benchmark_test_files.append(test_file)
        elif backend in str(test_file.name):
            session.config.benchmark_test_files.append(test_file)

    # Keeping track of all external test files and their paths
    session.config.external_test_files = {}
    if session.config.getoption("external_file_directory"):
        path_of_external_test_files = Path(
            session.config.getoption("external_file_directory")
        )
        external_files = sorted(path_of_external_test_files.glob("*"))
        for external_file in external_files:
            file_name = external_file.name
            session.config.external_test_files[file_name] = external_file


def pytest_sessionfinish(session, exitstatus):
    markdown_data = {
        "time_summary": [
            "Model name",
            "Submodel name",
            "Current time (ms)",
            "Expected/golden time (ms)",
        ],
        "dispatch_summary": [
            "Model name",
            "Submodel name",
            "Current dispatch count",
            "Expected/golden dispatch count",
        ],
        "size_summary": [
            "Model name",
            "Submodel name",
            "Current binary size (bytes)",
            "Expected/golden binary size (bytes)",
        ],
    }

    with open(f"{job_summary_path}/job_summary.md", "a") as job_summary, open(
        f"{job_summary_path}/job_summary.json", "r"
    ) as content:
        summary_data = json.loads(content.read())
        for key, value in markdown_data.items():
            if key in summary_data:
                table_data = tabulate.tabulate(
                    summary_data.get(key), headers=value, tablefmt="pipe"
                )
                print("\n" + table_data, file=job_summary)

    logger.info("Pytest benchmark test session has finished")


def pytest_collect_file(parent, file_path):
    # Run only the benchmark test for this directory
    if "model_benchmark_run" in str(file_path):
        return SharkTankModelBenchmarkTests.from_parent(parent, path=file_path)


@dataclass(frozen=True)
class BenchmarkTestSpec:
    model_name: str
    benchmark_file_name: str
    file_path: Path
    external_test_files: dict


class SharkTankModelBenchmarkTests(pytest.File):
    def collect(self):
        for file_path in self.config.benchmark_test_files:
            benchmark_file_name = file_path.stem
            model_name = str(file_path.parent.stem)

            item_name = f"{model_name} :: {benchmark_file_name}"

            spec = BenchmarkTestSpec(
                model_name=model_name,
                benchmark_file_name=benchmark_file_name,
                file_path=file_path,
                external_test_files=self.config.external_test_files,
            )

            yield ModelBenchmarkRunItem.from_parent(self, name=item_name, spec=spec)
