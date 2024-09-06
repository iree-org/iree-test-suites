# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import onnx
import pytest
import subprocess
import urllib.request
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent
ARTIFACTS_DIR = THIS_DIR / "artifacts"

ONNX_CONVERTER_OUTPUT_MIN_VERSION = 17


# TODO(#18289): use real frontend API, import model in-memory?
def upgrade_onnx_model(original_path: Path):
    original_model = onnx.load_model(original_path)
    converted_model = onnx.version_converter.convert_version(
        original_model, ONNX_CONVERTER_OUTPUT_MIN_VERSION
    )
    upgraded_path = original_path.with_name(
        original_path.stem + f"_version{ONNX_CONVERTER_OUTPUT_MIN_VERSION}.onnx"
    )
    logging.info(
        f"Upgrading '{original_path.relative_to(THIS_DIR)}' to '{upgraded_path.relative_to(THIS_DIR)}'"
    )
    onnx.save(converted_model, upgraded_path)
    return upgraded_path


# TODO(#18289): use real frontend API, import model in-memory?
def import_onnx_model(onnx_path: Path):
    imported_mlir_path = onnx_path.with_suffix(".mlir")
    logging.info(
        f"Importing '{onnx_path.relative_to(THIS_DIR)}' to '{imported_mlir_path.relative_to(THIS_DIR)}'"
    )
    exec_args = [
        "iree-import-onnx",
        str(onnx_path),
        "-o",
        str(imported_mlir_path),
    ]
    ret = subprocess.run(exec_args, capture_output=True)
    if ret.returncode != 0:
        logger.error(f"Import of '{onnx_path.name}' failed!\niree-import-onnx stdout:")
        logger.error(ret.stdout.decode("utf-8"))
        logger.error("iree-import-onnx stderr:")
        logger.error(ret.stderr.decode("utf-8"))
        raise RuntimeError(f"  '{onnx_path.name}' import failed")
    return imported_mlir_path


def compile_model(input_program: Path, config_name: str, compile_flags: List[str]):
    cwd = THIS_DIR
    compiled_module_path = input_program.with_name(
        input_program.stem + f"_{config_name}.vmfb"
    )
    compile_args = ["iree-compile", input_program.relative_to(cwd)]
    compile_args.extend(compile_flags)
    compile_args.extend(["-o", compiled_module_path.relative_to(cwd)])
    compile_cmd = subprocess.list2cmdline(compile_args)
    logging.getLogger().info(
        f"Launching compile command:\n"  #
        f"  cd {cwd} && {compile_cmd}"
    )
    ret = subprocess.run(compile_cmd, shell=True, capture_output=True, cwd=cwd)
    if ret.returncode != 0:
        logging.getLogger().error(f"Compilation of '{compiled_module_path}' failed")
        logger.error(ret.stdout.decode("utf-8"))
        logger.error("iree-import-onnx stderr:")
        logger.error(ret.stderr.decode("utf-8"))
        raise RuntimeError(f"  '{compiled_module_path.name}' compile failed")
    return compiled_module_path


def test_basic():
    print("test_basic")

    # TODO(scotttodd): move to fixture with cache / download on demand
    onnx_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
    original_path = ARTIFACTS_DIR / "mobilenetv2-12.onnx"
    # urllib.request.urlretrieve(onnx_url, original_path)

    upgraded_path = upgrade_onnx_model(original_path)
    imported_mlir_path = import_onnx_model(upgraded_path)
    compiled_module_path = compile_model(
        imported_mlir_path, "cpu", ["--iree-hal-target-backends=llvm-cpu"]
    )
    # TODO(scotttodd): Load input data
    # TODO(scotttodd): Run with IREE
    # TODO(scotttodd): Load into ONNX Runtime
    # TODO(scotttodd): Run with ONNX Runtime
    # TODO(scotttodd): Compare results
