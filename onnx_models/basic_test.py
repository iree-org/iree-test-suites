# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import numpy as np
import onnx
import struct
import pytest
import subprocess
import urllib.request
from onnxruntime import InferenceSession
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)
rng = np.random.default_rng(0)

THIS_DIR = Path(__file__).parent
ARTIFACTS_DIR = THIS_DIR / "artifacts"

ONNX_CONVERTER_OUTPUT_MIN_VERSION = 17


# TODO(#18289): use real frontend API, import model in-memory?
def upgrade_onnx_model_version(original_onnx_path: Path):
    original_model = onnx.load_model(original_onnx_path)
    converted_model = onnx.version_converter.convert_version(
        original_model, ONNX_CONVERTER_OUTPUT_MIN_VERSION
    )
    upgraded_onnx_path = original_onnx_path.with_name(
        original_onnx_path.stem + f"_version{ONNX_CONVERTER_OUTPUT_MIN_VERSION}.onnx"
    )
    logging.info(
        f"Upgrading '{original_onnx_path.relative_to(THIS_DIR)}' to '{upgraded_onnx_path.relative_to(THIS_DIR)}'"
    )
    onnx.save(converted_model, upgraded_onnx_path)
    return upgraded_onnx_path


# TODO(#18289): use real frontend API, import model in-memory?
def import_onnx_model_to_mlir(onnx_path: Path):
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
        logger.error(f"Import of '{onnx_path.name}' failed!")
        logger.error("iree-import-onnx stdout:")
        logger.error(ret.stdout.decode("utf-8"))
        logger.error("iree-import-onnx stderr:")
        logger.error(ret.stderr.decode("utf-8"))
        raise RuntimeError(f"  '{onnx_path.name}' import failed")
    return imported_mlir_path


def compile_mlir_with_iree(mlir_path: Path, config_name: str, compile_flags: List[str]):
    cwd = THIS_DIR
    iree_module_path = mlir_path.with_name(mlir_path.stem + f"_{config_name}.vmfb")
    compile_args = ["iree-compile", mlir_path.relative_to(cwd)]
    compile_args.extend(compile_flags)
    compile_args.extend(["-o", iree_module_path.relative_to(cwd)])
    compile_cmd = subprocess.list2cmdline(compile_args)
    logger.info(
        f"Launching compile command:\n"  #
        f"  cd {cwd} && {compile_cmd}"
    )
    ret = subprocess.run(compile_cmd, shell=True, capture_output=True, cwd=cwd)
    if ret.returncode != 0:
        logger.error(f"Compilation of '{iree_module_path}' failed")
        logger.error("iree-compile stdout:")
        logger.error(ret.stdout.decode("utf-8"))
        logger.error("iree-compile stderr:")
        logger.error(ret.stderr.decode("utf-8"))
        raise RuntimeError(f"  '{iree_module_path.name}' compile failed")
    return iree_module_path


def run_iree_module(iree_module_path: Path, run_flags: List[str]):
    cwd = THIS_DIR
    run_args = ["iree-run-module", f"--module={iree_module_path.relative_to(cwd)}"]
    run_args.extend(run_flags)
    run_cmd = subprocess.list2cmdline(run_args)
    logger.info(
        f"Launching run command:\n"  #
        f"  cd {cwd} && {run_cmd}"
    )
    ret = subprocess.run(run_cmd, shell=True, capture_output=True, cwd=cwd)
    if ret.returncode != 0:
        logger.error(f"Run of '{iree_module_path}' failed")
        logger.error("iree-run-module stdout:")
        logger.error(ret.stdout.decode("utf-8"))
        logger.error("iree-run-module stderr:")
        logger.error(ret.stderr.decode("utf-8"))
        raise RuntimeError(f"  '{iree_module_path.name}' run failed")
    # TODO(scotttodd): write outputs to files, or use --expected_output
    logger.info(f"Run of '{iree_module_path}' succeeded")
    logger.info("iree-run-module stdout:")
    logger.info(ret.stdout.decode("utf-8"))
    logger.info("iree-run-module stderr:")
    logger.info(ret.stderr.decode("utf-8"))


# map numpy dtype -> (iree dtype, struct.pack format str)
numpy_to_iree_dtype_map = {
    np.dtype("int64"): ("si64", "q"),
    np.dtype("uint64"): ("ui64", "Q"),
    np.dtype("int32"): ("si32", "i"),
    np.dtype("uint32"): ("ui32", "I"),
    np.dtype("int16"): ("si16", "h"),
    np.dtype("uint16"): ("ui16", "H"),
    np.dtype("int8"): ("si8", "b"),
    np.dtype("uint8"): ("ui8", "B"),
    np.dtype("float64"): ("f64", "d"),
    np.dtype("float32"): ("f32", "f"),
    np.dtype("float16"): ("f16", "e"),
    np.dtype("bool"): ("i1", "?"),
}


def pack_ndarray_to_binary(ndarr: np.ndarray):
    mylist = ndarr.flatten().tolist()
    dtype = ndarr.dtype
    bytearr = b""
    if dtype in numpy_to_iree_dtype_map:
        iree_dtype = numpy_to_iree_dtype_map[dtype][1]
        bytearr = struct.pack(f"{len(mylist)}{iree_dtype}", *mylist)
    else:
        raise NotImplementedError(
            f"Unsupported data type in pack_ndarray_to_binary(): '{dtype}'"
        )
    return bytearr


def write_binary_to_file(ndarr: np.ndarray, filename: Path):
    with open(filename, "wb") as f:
        bytearr = pack_ndarray_to_binary(ndarr)
        f.write(bytearr)


def test_basic():
    if not ARTIFACTS_DIR.is_dir():
        ARTIFACTS_DIR.mkdir(parents=True)

    # TODO(scotttodd): move to fixture with cache / download on demand
    # onnx_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
    # original_onnx_path = ARTIFACTS_DIR / "mobilenetv2-12.onnx"
    onnx_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx"
    original_onnx_path = ARTIFACTS_DIR / "resnet50-v1-12.onnx"
    # urllib.request.urlretrieve(onnx_url, original_onnx_path)

    upgraded_onnx_path = upgrade_onnx_model_version(original_onnx_path)
    imported_mlir_path = import_onnx_model_to_mlir(upgraded_onnx_path)
    iree_module_path = compile_mlir_with_iree(
        imported_mlir_path, "cpu", ["--iree-hal-target-backends=llvm-cpu"]
    )

    # # TODO(scotttodd): prepare_input helper function
    random_data = rng.random((1, 3, 224, 224), dtype=np.float32)
    random_data_path = original_onnx_path.with_name(
        original_onnx_path.stem + "_input_0.bin"
    )
    write_binary_to_file(random_data, random_data_path)
    # logger.info(random_data)

    run_iree_module(
        iree_module_path,
        ["--device=local-task", f"--input=1x3x224x224xf32=@{random_data_path}"],
    )

    onnx_session = InferenceSession(upgraded_onnx_path)
    # onnx_results = onnx_session.run(["output"], {"input": random_data})
    onnx_results = onnx_session.run(["resnetv17_dense0_fwd"], {"data": random_data})
    logger.info(onnx_results)

    # TODO(scotttodd): Compare results


# What varies between each test:
#   Model URL
#   Model name
#   Function signature
#     Number of inputs
#     Names of inputs
#     Shapes of inputs
#     Number of outputs
#     Names of outputs
#     Shapes of outputs
