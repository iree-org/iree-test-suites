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
    parser.add_argument("--submodel", type=str, default="*")
    parser.add_argument("--sku", type=str, default="mi300")
    parser.add_argument("--backend", type=str, default="gfx942")
    args = parser.parse_args()
    model = args.model
    submodel = args.submodel
    sku = args.sku
    backend = args.backend

    os.environ["THRESHOLD_MODEL"] = model
    os.environ["THRESHOLD_SUBMODEL"] = submodel
    os.environ["SKU"] = sku
    os.environ["BACKEND"] = backend

    THIS_DIR = Path(__file__).parent

    command = [
        "pytest",
        THIS_DIR / "test_model_threshold.py",
        "-rpFe",
        "--log-cli-level=info",
        "--capture=no",
        "--timeout=600",
        "--durations=0",
    ]
    ret_value = subprocess.run(command)
    return ret_value.returncode


if __name__ == "__main__":
    sys.exit(main())
