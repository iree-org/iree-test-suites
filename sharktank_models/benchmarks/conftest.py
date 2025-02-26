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

def pytest_sessionstart(session):
    with open("job_summary.md", "a") as job_summary, open("job_summary.json", "w+") as content:
        print(f"{sku.upper()} Complete Benchmark Summary:\n", file=job_summary)
        json.dump({}, content)
        
    logger.info("Pytest benchmark test session is starting")
    
def pytest_sessionfinish(session, exitstatus):
    markdown_data = {
        "time_summary": ["Model name", "Submodel name", "Current time (ms)", "Expected/golden time (ms)"],
        "dispatch_summary": ["Model name", "Submodel name", "Current dispatch count", "Expected/golden dispatch count"],
        "size_summary": ["Model name", "Submodel name", "Current binary size (bytes)", "Expected/golden binary size (bytes)"]
    }
    
    with open("job_summary.md", "a") as job_summary, open("job_summary.json", "r") as content:
        summary_data = json.loads(content.read())
        for key, value in markdown_data.items():
            if key in summary_data:
                table_data = tabulate.tabulate(
                    summary_data.get(key), headers=value, tablefmt="pipe"
                )
                print("\n" + table_data, file=job_summary)
    
    logger.info("Pytest benchmark test session has finished")

def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".json" and "job_summary" not in file_path.name and "benchmarks" in str(THIS_DIR):
        return SharkTankModelBenchmarkTests.from_parent(parent, path=file_path)

@dataclass(frozen = True)
class BenchmarkTestSpec:
    model_name: str
    benchmark_file_name: str
    
class SharkTankModelBenchmarkTests(pytest.File):
    
    def collect(self):
        path = str(self.path).split("/")
        benchmark_file_name = path[-1].replace(".json", "")
        model_name = path[-2]
        
        item_name = f"{model_name} :: {benchmark_file_name}"
        
        spec = BenchmarkTestSpec(
            model_name = model_name,
            benchmark_file_name = benchmark_file_name
        )
        
        yield ModelBenchmarkRunItem.from_parent(self, name=item_name, spec=spec)
    