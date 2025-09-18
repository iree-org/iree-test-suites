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
import subprocess

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

    parser.addoption(
        "--export",
        action="store_true",
        default=False,
        help="If set, will export ir from sharktank before running benchmarks",
    )


def pytest_configure():
    pytest.vmfb_manager = {}


def pytest_sessionstart(session):
    logger.info("Pytest quality test session is starting")


    # Export first from sharktank if flag is passed
    if session.config.getoption("export"):
        run_export_first()

    # Collect all .json files for quality tests
    session.config.quality_test_files = []
    path_of_quality_tests = Path(session.config.getoption("test_file_directory"))
    test_files = sorted(path_of_quality_tests.glob("**/*.json"))
    for test_file in test_files:
        if backend in str(test_file.name):
            session.config.quality_test_files.append(test_file)

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

# Function for running Export script
def run_export_first():
    logger.info("Running export step...")

    script_path = THIS_DIR / "run_export.py"
    try:
        subprocess.run(
            ["python3", str(script_path)],
            check=True,
        )
        logger.info("Export finished successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Export failed: {e}")
        pytest.exit("Stopping pytest because export failed.", returncode=1)

# def run_cmd(cmd, log_file):
#     log_path = OUTPUT_DIR / log_file
#     log_path.parent.mkdir(parents=True, exist_ok=True)

#     with open(log_path, "w") as f:
#         process = subprocess.Popen(
#             cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
#         )
#         for line in process.stdout:
#             decoded = line.decode()
#             f.write(decoded)
#             print(decoded, end="")
#         process.wait()
#         if process.returncode != 0:
#             raise RuntimeError(f"Command failed: {cmd}")
#     return log_path

# def run_export_first():
#     logger.info("Running export...")

#     gen_mlir_path = f"{model_config['output_dir']}/output.mlir"
#     gen_config_path = f"{model_config['output_dir']}/config_attn.json"

#     # if os.path.exists(gen_mlir_path) and os.path.exists(gen_config_path):
#     #     logger.info("File exists. Skipping Export....")
#     #     return

#     logger.info("Continuing With Export...")

#     cmd = (
#         f"python scripts/run_export.py --irpa {model_config['irpa']} "
#         f"--attention-kernel {model_config['attention_kernel']} "
#         f"--dtype {model_config['dtype']} "
#         f"--bs-prefill {model_config['bs_prefill']} "
#         f"--bs-decode {model_config['bs_decode']} "
#         f"--device-block-count {model_config['device_block_count']} "
#         f"--extra-export-flags-list \"{model_config['extra_export_flags_list']}\" "
#         f"--output-dir {model_config['output_dir']}"
#     )

#     try:
#         log_path = run_cmd(cmd, "export.log")
#         logger.info(f"Export completed successfully. Log saved at {log_path}")
#     except Exception as e:
#         logger.error(f"Export failed: {e}")
#         pytest.exit("Stopping pytest because export failed.", returncode=1)


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
