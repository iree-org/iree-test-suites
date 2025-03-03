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
    with open("job_summary.md", "a") as job_summary, open(
        "job_summary.json", "w+"
    ) as content:
        print(f"{sku.upper()} Complete Benchmark Summary:\n", file=job_summary)
        json.dump({}, content)

    # Collect all .json files for benchmark tests
    session.config.benchmark_test_files = []
    path_of_benchmark_tests = session.config.getoption("test_file_directory")
    if path_of_benchmark_tests:
        for root, dirs, files in os.walk(path_of_benchmark_tests):
            for file in files:
                path_of_file = str(os.path.join(root, file))
                if ".json" in path_of_file:
                    session.config.benchmark_test_files.append(path_of_file)

    # Keeping track of all external test files and their paths
    session.config.external_test_files = {}
    path_of_external_test_files = session.config.getoption("external_file_directory")
    if path_of_external_test_files:
        for root, dirs, files in os.walk(path_of_external_test_files):
            for file in files:
                path_of_file = str(os.path.join(root, file))
                file_name = path_of_file.split("/")[-1]
                session.config.external_test_files[file_name] = path_of_file


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

    with open("job_summary.md", "a") as job_summary, open(
        "job_summary.json", "r"
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
    file_path: str
    external_test_files: dict


class SharkTankModelBenchmarkTests(pytest.File):
    def collect(self):
        for file_path in self.config.benchmark_test_files:
            path = file_path.split("/")
            benchmark_file_name = path[-1].replace(".json", "")
            model_name = path[-2]

            item_name = f"{model_name} :: {benchmark_file_name}"

            spec = BenchmarkTestSpec(
                model_name=model_name,
                benchmark_file_name=benchmark_file_name,
                file_path=file_path,
                external_test_files=self.config.external_test_files,
            )

            yield ModelBenchmarkRunItem.from_parent(self, name=item_name, spec=spec)
