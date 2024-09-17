# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import numpy as np
import onnx
import struct
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent

###############################################################################
# Exception types
###############################################################################


class IreeImportOnnxException(RuntimeError):
    pass


class IreeCompileException(RuntimeError):
    pass


class IreeRunException(RuntimeError):
    pass


###############################################################################
# Numpy utilities
###############################################################################


def write_ndarray_to_binary_file(ndarr: np.ndarray, filename: Path):
    with open(filename, "wb") as f:
        bytearr = pack_ndarray_to_binary(ndarr)
        f.write(bytearr)


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


###############################################################################
# ONNX utilities
###############################################################################


def convert_proto_elem_type_to_iree_dtype(etype) -> str:
    if etype == onnx.TensorProto.BOOL:
        return "i1"
    if etype == onnx.TensorProto.INT4 or etype == onnx.TensorProto.UINT4:
        return "i4"
    if etype == onnx.TensorProto.INT8 or etype == onnx.TensorProto.UINT8:
        return "i8"
    if etype == onnx.TensorProto.INT16 or etype == onnx.TensorProto.UINT16:
        return "i16"
    if etype == onnx.TensorProto.INT32 or etype == onnx.TensorProto.UINT32:
        return "i32"
    if etype == onnx.TensorProto.INT64 or etype == onnx.TensorProto.UINT64:
        return "i64"
    if etype == onnx.TensorProto.FLOAT16:
        return "f16"
    if etype == onnx.TensorProto.FLOAT:
        return "f32"
    if etype == onnx.TensorProto.DOUBLE:
        return "f64"
    if etype == onnx.TensorProto.COMPLEX64:
        return "complex<f32>"
    if etype == onnx.TensorProto.COMPLEX128:
        return "complex<f64>"
    if etype == onnx.TensorProto.BFLOAT16:
        return "bf16"
    if etype == onnx.TensorProto.FLOAT8E4M3FN:
        return "f8e4m3fn"
    if etype == onnx.TensorProto.FLOAT8E4M3FNUZ:
        return "f8e4m3fnuz"
    if etype == onnx.TensorProto.FLOAT8E5M2:
        return "f8e5m2"
    if etype == onnx.TensorProto.FLOAT8E5M2FNUZ:
        return "f8e5m2fnuz"
    return ""


# TODO(#18289): use real frontend API, import model in-memory?
def upgrade_onnx_model_version(original_onnx_path: Path, min_version=17):
    original_model = onnx.load_model(original_onnx_path)
    original_version = original_model.opset_import[0].version
    if original_version >= min_version:
        logger.debug(
            f"ONNX model at {original_onnx_path.relative_to(THIS_DIR)} version {original_version} >= {min_version}, skipping upgrade"
        )
        return original_onnx_path

    converted_model = onnx.version_converter.convert_version(
        original_model, min_version
    )
    upgraded_onnx_path = original_onnx_path.with_name(
        original_onnx_path.stem + f"_version{min_version}.onnx"
    )
    logger.info(
        f"Upgrading '{original_onnx_path.relative_to(THIS_DIR)}' to '{upgraded_onnx_path.relative_to(THIS_DIR)}'"
    )
    onnx.save(converted_model, upgraded_onnx_path)
    return upgraded_onnx_path


# TODO(#18289): use real frontend API, import model in-memory?
def import_onnx_model_to_mlir(onnx_path: Path):
    imported_mlir_path = onnx_path.with_suffix(".mlir")
    logger.info(
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
        raise IreeImportOnnxException(f"  '{onnx_path.name}' import failed")
    return imported_mlir_path
