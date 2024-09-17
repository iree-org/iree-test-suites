# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import numpy as np
import onnx
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
# ONNX loading, running, import, etc.
###############################################################################


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
        write_ndarray_to_binary_file(input_data, input_data_path)

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
            write_ndarray_to_binary_file(onnx_result, reference_output_data_path)
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
