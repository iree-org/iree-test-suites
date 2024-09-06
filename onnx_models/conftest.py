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
from onnx.mapping import TENSOR_TYPE_MAP
from onnxruntime import InferenceSession
from pathlib import Path
from typing import List

from .utils import *

logger = logging.getLogger(__name__)
rng = np.random.default_rng(0)

THIS_DIR = Path(__file__).parent
ARTIFACTS_DIR = THIS_DIR / "artifacts"

###############################################################################
# General utilities
###############################################################################

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


###############################################################################
# ONNX loading, running, import, etc.
###############################################################################

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


def convert_onnx_tensor_proto_to_numpy_dimensions(
    tensor_proto: onnx.onnx_ml_pb2.TensorProto,
) -> str:
    # Note: turning dynamic dimensions into just 1 here, since we need
    # a concrete (static) shape buffer of input data in the tests.
    return tuple(
        d.dim_value if d.HasField("dim_value") else 1 for d in tensor_proto.shape.dim
    )


def convert_onnx_tensor_proto_to_iree_type_string(
    tensor_proto: onnx.onnx_ml_pb2.TensorProto,
) -> str:
    shape = tensor_proto.shape
    # Note: turning dynamic dimensions into just "1" here, since we need
    # a concrete (static) shape buffer of input data in the tests.
    shape = "x".join(
        [str(d.dim_value) if d.HasField("dim_value") else "1" for d in shape.dim]
    )
    dtype = convert_proto_elem_type_to_iree_dtype(tensor_proto.elem_type)
    if shape == "":
        return dtype
    return f"{shape}x{dtype}"


def get_onnx_model_metadata(onnx_path: Path):
    # We can either
    #   A) List all metadata explicitly
    #   B) Get metadata on demand from the .onnx protobuf using 'onnx'
    #   C) Get metadata on demand from the InferenceSession using 'onnxruntime'
    # This is option B.

    logger.info(f"Getting model metadata for '{onnx_path.relative_to(THIS_DIR)}'")
    model = onnx.load(onnx_path)

    inputs = []
    for idx, graph_input in enumerate(model.graph.input):
        type_proto = graph_input.type
        if not type_proto.HasField("tensor_type"):
            raise NotImplementedError(f"Unsupported proto type: {type_proto}")
        tensor_type = type_proto.tensor_type

        # Create a numpy tensor with some random data for the input.
        numpy_dimensions = convert_onnx_tensor_proto_to_numpy_dimensions(tensor_type)
        numpy_dtype = TENSOR_TYPE_MAP[tensor_type.elem_type].np_dtype
        if numpy_dtype == np.float32 or numpy_dtype == np.float64:
            input_data = rng.random(numpy_dimensions, dtype=numpy_dtype)
        elif numpy_dtype == np.int32 or numpy_dtype == np.int64:
            input_data = rng.integers(numpy_dimensions, dtype=numpy_dtype)
        else:
            raise NotImplementedError(f"Unsupported numpy type: {numpy_dtype}")
        logger.debug(input_data)
        input_data_path = onnx_path.with_name(onnx_path.stem + f"_input_{idx}.bin")
        write_binary_to_file(input_data, input_data_path)

        iree_type = convert_onnx_tensor_proto_to_iree_type_string(tensor_type)
        inputs.append(
            {
                "name": graph_input.name,
                "iree_type": iree_type,
                "input_data": input_data,
                "input_data_path": input_data_path,
            }
        )

    outputs = []
    for graph_output in model.graph.output:
        type_proto = graph_output.type
        if not type_proto.HasField("tensor_type"):
            raise NotImplementedError(f"Unsupported proto type: {type_proto}")
        tensor_type = type_proto.tensor_type

        iree_type = convert_onnx_tensor_proto_to_iree_type_string(tensor_type)
        outputs.append(
            {
                "name": graph_output.name,
                "iree_type": iree_type,
            }
        )

    return {
        "inputs": inputs,
        "outputs": outputs,
    }


###############################################################################
# IREE compilation and running
###############################################################################


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
        raise IreeCompileException(f"  '{iree_module_path.name}' compile failed")
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
        raise IreeRunException(f"  '{iree_module_path.name}' run failed")


@pytest.fixture
def compare_between_iree_and_onnxruntime():
    def fn(
        model_name: str,
        model_url: str,
    ):
        if not ARTIFACTS_DIR.is_dir():
            ARTIFACTS_DIR.mkdir(parents=True)
        # TODO(scotttodd): group model artifacts into subfolders

        # TODO(scotttodd): move to fixture with cache / download on demand
        # TODO(scotttodd): extract name from URL?
        # TODO(scotttodd): overwrite if already existing? check SHA?
        original_onnx_path = ARTIFACTS_DIR / f"{model_name}.onnx"
        if not original_onnx_path.exists():
            urllib.request.urlretrieve(model_url, original_onnx_path)

        # TODO(scotttodd): cache ONNX metadata and runtime results
        upgraded_onnx_path = upgrade_onnx_model_version(original_onnx_path)

        onnx_model_metadata = get_onnx_model_metadata(upgraded_onnx_path)
        logger.debug("ONNX model metadata:")
        logger.debug(onnx_model_metadata)

        # Run through ONNX Runtime.
        onnx_session = InferenceSession(upgraded_onnx_path)
        output_names = [output["name"] for output in onnx_model_metadata["outputs"]]
        inputs = {}
        for input in onnx_model_metadata["inputs"]:
            inputs[input["name"]] = input["input_data"]
        onnx_results = onnx_session.run(output_names, inputs)

        # Prepare inputs and expected outputs for running through IREE.
        run_module_args = []
        for input in onnx_model_metadata["inputs"]:
            input_type = input["iree_type"]
            input_data_path = input["input_data_path"]
            run_module_args.append(f"--input={input_type}=@{input_data_path}")

        assert len(onnx_model_metadata["outputs"]) == len(onnx_results)
        for idx in range(len(onnx_results)):
            output = onnx_model_metadata["outputs"][idx]
            output_type = output["iree_type"]
            onnx_result = onnx_results[idx]
            logger.debug(np.array(onnx_result))
            reference_output_data_path = original_onnx_path.with_name(
                original_onnx_path.stem + f"_output_{idx}.bin"
            )
            write_binary_to_file(onnx_result, reference_output_data_path)
            run_module_args.append(
                f"--expected_output={output_type}=@{reference_output_data_path}"
            )

        # Import, compile, then run with IREE.
        imported_mlir_path = import_onnx_model_to_mlir(upgraded_onnx_path)
        iree_module_path = compile_mlir_with_iree(
            imported_mlir_path, "cpu", ["--iree-hal-target-backends=llvm-cpu"]
        )
        # Note: could load the output into memory here and compare using numpy.
        run_flags = ["--device=local-task"]
        run_flags.extend(run_module_args)
        run_iree_module(iree_module_path, run_flags)

    return fn
