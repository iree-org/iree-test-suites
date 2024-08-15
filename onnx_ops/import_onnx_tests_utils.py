# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import onnx
import struct
from onnx import numpy_helper
from pathlib import Path


def convert_onnx_proto_to_numpy_array(
    proto_path: Path, type_proto: onnx.onnx_ml_pb2.TypeProto
) -> np.ndarray:
    with open(proto_path, "rb") as f:
        protobuf_content = f.read()
        if type_proto.HasField("tensor_type"):
            tensor = onnx.TensorProto()
            tensor.ParseFromString(protobuf_content)
            return numpy_helper.to_array(tensor)
        else:
            raise NotImplementedError(f"Unsupported proto type: {type_proto}")


def convert_proto_elem_type_to_iree_dtype(etype):
    if etype == onnx.TensorProto.FLOAT:
        return "f32"
    if etype == onnx.TensorProto.UINT8:
        return "i8"
    if etype == onnx.TensorProto.INT8:
        return "i8"
    if etype == onnx.TensorProto.UINT16:
        return "i16"
    if etype == onnx.TensorProto.INT16:
        return "i16"
    if etype == onnx.TensorProto.INT32:
        return "i32"
    if etype == onnx.TensorProto.INT64:
        return "i64"
    if etype == onnx.TensorProto.BOOL:
        return "i1"
    if etype == onnx.TensorProto.FLOAT16:
        return "f16"
    if etype == onnx.TensorProto.DOUBLE:
        return "f64"
    if etype == onnx.TensorProto.UINT32:
        return "i32"
    if etype == onnx.TensorProto.UINT64:
        return "i64"
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
    if etype == onnx.TensorProto.UINT4:
        return "i4"
    if etype == onnx.TensorProto.INT4:
        return "i4"
    return ""


def convert_onnx_type_proto_to_iree_type_string(
    type_proto: onnx.onnx_ml_pb2.TypeProto,
) -> str:
    if type_proto.HasField("tensor_type"):
        tensor_type = type_proto.tensor_type
        shape = tensor_type.shape
        shape = "x".join([str(d.dim_value) for d in shape.dim])
        dtype = convert_proto_elem_type_to_iree_dtype(tensor_type.elem_type)
        if shape == "":
            return dtype
        return f"{shape}x{dtype}"
    else:
        raise NotImplementedError(f"Unsupported proto type: {type_proto}")


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
