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


def convert_onnx_type_proto_to_numpy_dimensions(
    type_proto: onnx.onnx_ml_pb2.TypeProto,
) -> str:
    if type_proto.HasField("tensor_type"):
        # Note: turning dynamic dimensions into just 1 here, since we need
        # a concrete (static) shape buffer of input data in the tests.
        return tuple(
            d.dim_value if d.HasField("dim_value") else 1
            for d in type_proto.tensor_type.shape.dim
        )
    else:
        raise NotImplementedError(f"Unsupported proto type: {type_proto}")


def convert_onnx_type_proto_to_iree_type_string(
    type_proto: onnx.onnx_ml_pb2.TypeProto,
) -> str:
    if type_proto.HasField("tensor_type"):
        tensor_type = type_proto.tensor_type
        shape = tensor_type.shape
        # Note: turning dynamic dimensions into just "1" here, since we need
        # a concrete (static) shape buffer of input data in the tests.
        shape = "x".join(
            [str(d.dim_value) if d.HasField("dim_value") else "1" for d in shape.dim]
        )
        dtype = convert_proto_elem_type_to_iree_dtype(tensor_type.elem_type)
        if shape == "":
            return dtype
        return f"{shape}x{dtype}"
    else:
        raise NotImplementedError(f"Unsupported proto type: {type_proto}")


def get_onnx_model_metadata(onnx_path: Path):
    logger.info(f"Getting model metadata for '{onnx_path.relative_to(THIS_DIR)}'")
    model = onnx.load(onnx_path)

    inputs = []
    outputs = []

    # input_data = rng.random(input_shape, dtype=np.float32)
    # input_data_path = original_onnx_path.with_name(
    #     original_onnx_path.stem + "_input_0.bin"
    # )
    # write_binary_to_file(input_data, input_data_path)
    # logger.debug(input_data)

    # help(model.graph.input)
    # print(model.graph.input)
    for graph_input in model.graph.input:
        numpy_dimensions = convert_onnx_type_proto_to_numpy_dimensions(graph_input.type)
        iree_type = convert_onnx_type_proto_to_iree_type_string(graph_input.type)
        inputs.append(
            {
                "name": graph_input.name,
                "numpy_dimensions": numpy_dimensions,
                "iree_type": iree_type,
            }
        )
    for graph_output in model.graph.output:
        numpy_dimensions = convert_onnx_type_proto_to_numpy_dimensions(
            graph_output.type
        )
        iree_type = convert_onnx_type_proto_to_iree_type_string(graph_output.type)
        outputs.append(
            {
                "name": graph_output.name,
                "numpy_dimensions": numpy_dimensions,
                "iree_type": iree_type,
            }
        )

    # Concrete shape to generate test data with
    # (N, 3, 224, 224), with dynamic dim "N" should turn into
    # (1, 3, 224, 224)

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
    # logger.info(f"Run of '{iree_module_path}' succeeded")
    # logger.info("iree-run-module stdout:")
    # logger.info(ret.stdout.decode("utf-8"))
    # logger.info("iree-run-module stderr:")
    # logger.info(ret.stderr.decode("utf-8"))


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
# Can get the function signature from the loaded ONNX model


@pytest.fixture
def compare_between_iree_and_onnxruntime():
    def fn(
        model_name: str,
        model_url: str,
        input_name: str,
        input_shape: tuple[int, ...],
        input_type: str,
        output_name: str,
        output_shape: tuple[int, ...],
        output_type: str,
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

        upgraded_onnx_path = upgrade_onnx_model_version(original_onnx_path)

        onnx_model_metadata = get_onnx_model_metadata(upgraded_onnx_path)
        logger.info(onnx_model_metadata)
        return

        # TODO(scotttodd): prepare_input helper function, multiple inputs
        # TODO(scotttodd): dtype from input_shape (or ONNX model reflection)
        input_data = rng.random(input_shape, dtype=np.float32)
        input_data_path = original_onnx_path.with_name(
            original_onnx_path.stem + "_input_0.bin"
        )
        write_binary_to_file(input_data, input_data_path)
        logger.debug(input_data)

        # Run through ONNX Runtime.
        onnx_session = InferenceSession(upgraded_onnx_path)

        # We can either
        #   A) List all metadata explicitly
        #   B) Get metadata on demand from the .onnx protobuf using 'onnx'
        #   C) Get metadata on demand from the InferenceSession using 'onnxruntime'
        inputs = onnx_session.get_inputs()
        logger.info("inputs")
        for input in inputs:
            logger.info(f"{input.name}, {input.shape}, {input.type}")
            # if input.is_tensor():
            #     logger.info(f"  input element type: {input.element_type}")
        outputs = onnx_session.get_outputs()
        logger.info("outputs")
        for output in outputs:
            logger.info(f"{output.name}, {output.shape}, {output.type}")
        # input[0] : data,                 ['N', 3, 224, 224], tensor(float)
        # output[0]: resnetv17_dense0_fwd, ['N', 1000],        tensor(float)

        # TODO(scotttodd): multiple inputs/outputs
        onnx_results = onnx_session.run([output_name], {input_name: input_data})
        logger.debug(np.array(onnx_results[0]))
        reference_output_data_path = original_onnx_path.with_name(
            original_onnx_path.stem + "_output_0.bin"
        )
        write_binary_to_file(onnx_results[0], reference_output_data_path)

        # Import, compile, then run with IREE.
        imported_mlir_path = import_onnx_model_to_mlir(upgraded_onnx_path)
        iree_module_path = compile_mlir_with_iree(
            imported_mlir_path, "cpu", ["--iree-hal-target-backends=llvm-cpu"]
        )
        # Note: could load the output into memory here and compare using numpy.
        # TODO(scotttodd): signature conversions from onnx/numpy to IREE
        run_iree_module(
            iree_module_path,
            [
                "--device=local-task",
                f"--input=1x3x224x224xf32=@{input_data_path}",
                f"--expected_output=1x1000xf32=@{reference_output_data_path}",
            ],
        )

    return fn
