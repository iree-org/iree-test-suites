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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sdxl")
    parser.add_argument("--filename", type=str, default="*")
    parser.add_argument("--sku", type=str, default="mi300")
    parser.add_argument("--rocm-chip", type=str, default="gfx942")
    args = parser.parse_args()
    model = args.model
    filename = args.filename
    sku = args.sku
    rocm_chip = args.rocm_chip

    os.environ["BENCHMARK_MODEL"] = model
    os.environ["BENCHMARK_FILE_NAME"] = filename
    os.environ["SKU"] = sku
    os.environ["ROCM_CHIP"] = rocm_chip

    THIS_DIR = Path(__file__).parent
    
    with open("job_summary.md", "a") as job_summary:
        print(f"{sku.upper()} {model.upper()} Complete Benchmark Summary:\n\n", file=job_summary)

    command = [
        "pytest",
        THIS_DIR / "test_model_benchmark.py",
        "--log-cli-level=info",
        "--timeout=600",
        "--retries=7",
    ]
    subprocess.run(command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
