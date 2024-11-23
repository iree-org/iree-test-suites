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
from onnxruntime import NodeArg
from pathlib import Path

logger = logging.getLogger(__name__)
rng = np.random.default_rng(0)

# Convert test cases to at least this version using The ONNX Version Converter.
ONNX_CONVERTER_OUTPUT_MIN_VERSION = 17

THIS_DIR = Path(__file__).parent

###############################################################################
# Exception types
###############################################################################

# Note: can mark tests as expected to fail at a specific stage with:
# @pytest.mark.xfail(raises=IreeImportOnnxException)
# @pytest.mark.xfail(raises=IreeCompileException)
# @pytest.mark.xfail(raises=IreeRunException)


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


def convert_numpy_to_iree_type_string(ndarr: np.ndarray):
    shape = "x".join(str(x) for x in ndarr.shape)
    dtype = numpy_to_iree_dtype_map[ndarr.dtype][0]
    if shape == "":
        return dtype
    return f"{shape}x{dtype}"


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


def convert_ort_shape_to_numpy_dimensions(
    node_arg: NodeArg,
) -> tuple[int]:
    # Note: turning dynamic dimensions into just 1 here, since we need
    # a concrete (static) shape buffer of input data in the tests.
    # TODO(scotttodd): allow this to be overriden as needed
    return tuple(x if isinstance(x, int) else 1 for x in node_arg.shape)


def convert_ort_type_to_numpy_dtype(node_arg: NodeArg):
    type_str = node_arg.type
    if type_str[0:6] != "tensor":
        raise TypeError(f"node: {node_arg} has unhandled non-tensor type '{type_str}'")
    dtype_str = type_str[7:-1]
    if dtype_str == "float":
        return np.dtype("float32")
    if dtype_str == "int" or dtype_str == "int32":
        return np.dtype("int32")
    if dtype_str == "int64":
        return np.dtype("int64")
    if dtype_str == "int8":
        return np.dtype("int8")
    if dtype_str == "uint8":
        return np.dtype("uint8")
    if dtype_str == "bool":
        return np.dtype("bool")
    raise NotImplementedError(f"type conversion for '{type_str}' not implemented")


def convert_ort_type_to_iree_dtype(node_arg: NodeArg) -> str:
    numpy_dtype = convert_ort_type_to_numpy_dtype(node_arg)
    return numpy_to_iree_dtype_map[numpy_dtype][0]


def convert_ort_to_iree_type(
    node_arg: NodeArg,
) -> str:
    # Note: turning dynamic dimensions into just "1" here, since we need
    # a concrete (static) shape buffer of input data in the tests.
    # TODO(scotttodd): allow this to be overriden as needed
    shape = "x".join([str(x) if isinstance(x, int) else "1" for x in node_arg.shape])
    dtype = convert_ort_type_to_iree_dtype(node_arg)
    if shape == "":
        return dtype
    return f"{shape}x{dtype}"


def generate_numpy_input_for_ort_node_arg(node_arg: NodeArg):
    numpy_dimensions = convert_ort_shape_to_numpy_dimensions(node_arg)
    numpy_type = convert_ort_type_to_numpy_dtype(node_arg).type

    if numpy_type == np.float32 or numpy_type == np.float64:
        return rng.random(numpy_dimensions, dtype=numpy_type)
    if numpy_type == np.int32 or numpy_type == np.int64:
        return rng.integers(numpy_dimensions, dtype=numpy_type)
    # TODO(scotttodd): test i8, bool, and other dtypes
    # if numpy_type == np.int8:
    #     return rng.integers(-127, 128, size=numpy_dimensions, dtype=numpy_type)

    raise NotImplementedError(f"Unsupported numpy type: {numpy_type}")


# TODO(#18289): use real frontend API, import model in-memory?
def import_onnx_model_to_mlir(onnx_path: Path):
    imported_mlir_path = onnx_path.with_suffix(".mlir")
    logger.info(
        f"Importing '{onnx_path.relative_to(THIS_DIR)}' to '{imported_mlir_path.relative_to(THIS_DIR)}'"
    )
    exec_args = [
        "iree-import-onnx",
        str(onnx_path),
        "--opset-version",
        str(ONNX_CONVERTER_OUTPUT_MIN_VERSION),
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
