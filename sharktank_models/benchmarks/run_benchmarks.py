# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
import os
from pathlib import Path
import argparse
import sys
import tabulate
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sdxl")
    parser.add_argument("--filename", type=str, default="*")
    parser.add_argument("--sku", type=str, default="mi300")
    parser.add_argument("--backend", type=str, default="gfx942")
    args = parser.parse_args()
    model = args.model
    filename = args.filename
    sku = args.sku
    backend = args.backend

    os.environ["BENCHMARK_MODEL"] = model
    os.environ["BENCHMARK_FILE_NAME"] = filename
    os.environ["SKU"] = sku
    os.environ["BACKEND"] = backend

    THIS_DIR = Path(__file__).parent

    command = [
        "pytest",
        THIS_DIR / "test_model_benchmark.py",
        "--log-cli-level=info",
        "--timeout=600",
        "--retries=0",
    ]
    
    with open("job_summary.md", "a") as job_summary, open("job_summary.json", "w+") as content:
        print(f"{sku.upper()} {model.upper()} Complete Benchmark Summary:\n", file=job_summary)
        json.dump({}, content)
    
    ret_value = subprocess.run(command)
    
    markdown_data = {
        "time_summary": ["Submodel name", "Current time (ms)", "Expected/golden time (ms)"],
        "dispatch_summary": ["Submodel name", "Current dispatch count", "Expected/golden dispatch count"],
        "size_summary": ["Submodel name", "Current binary size (bytes)", "Expected/golden binary size (bytes)"]
    }
    
    with open("job_summary.md", "a") as job_summary, open("job_summary.json", "r") as content:
        summary_data = json.loads(content.read())
        for key, value in markdown_data.items():
            if key in summary_data:
                print("\n" + f"golden {key}" + "\n")
                table_data = tabulate.tabulate(
                    summary_data.get(key), headers=value, tablefmt="pipe"
                )
                print("\n" + table_data, file=job_summary)
            
    return ret_value.returncode


if __name__ == "__main__":
    sys.exit(main())
