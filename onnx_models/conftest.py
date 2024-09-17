# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import numpy as np
import pytest
import subprocess
import urllib.request
from dataclasses import dataclass
from onnxruntime import InferenceSession, NodeArg
from pathlib import Path

from .utils import *

logger = logging.getLogger(__name__)
rng = np.random.default_rng(0)

THIS_DIR = Path(__file__).parent
ARTIFACTS_ROOT = THIS_DIR / "artifacts"


###############################################################################
# ONNX loading, running, import, etc.
###############################################################################


@dataclass(frozen=True)
class IreeModelParameterMetadata:
    """Metadata for a single input or output used with iree-run-module tooling.

    Args:
        name: The name of the parameter.
        type: The type of the parameter as expected by the tools, e.g. "2x2xi32".
        data_file: Path to either the input or expected output binary file for this parameter.
    """

    name: str
    type: str
    data_file: Path


@dataclass(frozen=True)
class OnnxModelMetadata:
    """Metadata for an ONNX model.

    Args:
        inputs: One parameter metadata per input.
        outputs: One parameter metadata per output.
    """

    inputs: list[IreeModelParameterMetadata]
    outputs: list[IreeModelParameterMetadata]


def convert_onnxruntime_node_arg_to_numpy_dimensions(
    node_arg: NodeArg,
) -> tuple[int]:
    # Note: turning dynamic dimensions into just 1 here, since we need
    # a concrete (static) shape buffer of input data in the tests.
    return tuple(x if isinstance(x, int) else 1 for x in node_arg.shape)


def convert_onnxruntime_shape_to_iree_type_string(
    node_arg: NodeArg,
) -> str:
    # Note: turning dynamic dimensions into just "1" here, since we need
    # a concrete (static) shape buffer of input data in the tests.
    shape = "x".join([str(x) if isinstance(x, int) else "1" for x in node_arg.shape])
    dtype = convert_node_arg_type_to_iree_dtype(node_arg.type)
    if shape == "":
        return dtype
    return f"{shape}x{dtype}"


def get_onnx_model_metadata(onnx_path: Path) -> OnnxModelMetadata:
    # We can either
    #   A) List all metadata explicitly
    #   B) Get metadata on demand from the .onnx protobuf using 'onnx'
    #   C) Get metadata on demand from the InferenceSession using 'onnxruntime'
    # This is option C.

    onnx_session = InferenceSession(onnx_path)
    logger.info(f"Getting model metadata for '{onnx_path.relative_to(THIS_DIR)}'")
    inputs = []
    onnx_inputs = {}
    for idx, input in enumerate(onnx_session.get_inputs()):
        logger.debug(f"Session input [{idx}]")
        logger.debug(f"  name: '{input.name}'")
        numpy_dimensions = convert_onnxruntime_node_arg_to_numpy_dimensions(input)
        iree_type = convert_onnxruntime_shape_to_iree_type_string(input)
        logger.debug(f"  shape: {input.shape}")
        logger.debug(f"  numpy shape: {numpy_dimensions}")
        logger.debug(f"  type: '{input.type}'")
        logger.debug(f"  iree parameter: {iree_type}")

        # Create a numpy tensor with some random data for the input.
        numpy_dtype = convert_node_arg_type_to_numpy_dtype(input.type)
        if numpy_dtype == np.float32 or numpy_dtype == np.float64:
            input_data = rng.random(numpy_dimensions, dtype=numpy_dtype)
        elif numpy_dtype == np.int32 or numpy_dtype == np.int64:
            input_data = rng.integers(numpy_dimensions, dtype=numpy_dtype)
        else:
            raise NotImplementedError(f"Unsupported numpy type: {numpy_dtype}")
        input_data_path = onnx_path.with_name(onnx_path.stem + f"_input_{idx}.bin")
        write_ndarray_to_binary_file(input_data, input_data_path)

        inputs.append(
            IreeModelParameterMetadata(
                name=input.name,
                type=iree_type,
                data_file=input_data_path,
            )
        )
        onnx_inputs[input.name] = input_data

    # Run through onnxruntime and then save the output results.
    output_names = [output.name for output in onnx_session.get_outputs()]
    onnx_results = onnx_session.run(output_names, onnx_inputs)

    assert len(onnx_session.get_outputs()) == len(onnx_results)
    outputs = []
    for i in range(len(onnx_results)):
        output = onnx_session.get_outputs()[i]
        result = onnx_results[i]
        logger.debug(f"Session output [{idx}]")
        logger.debug(f"  name: '{output.name}'")
        logger.debug(f"  shape (actual): {result.shape}")
        logger.debug(f"  type (numpy): '{result.dtype}'")
        iree_type = convert_numpy_to_iree_type_string(result)
        output_data_path = onnx_path.with_name(onnx_path.stem + f"_output_{idx}.bin")
        write_ndarray_to_binary_file(result, output_data_path)

        outputs.append(
            IreeModelParameterMetadata(
                name=output.name,
                type=iree_type,
                data_file=output_data_path,
            )
        )

    return OnnxModelMetadata(inputs=inputs, outputs=outputs)


###############################################################################
# IREE compilation and running
###############################################################################


def compile_mlir_with_iree(mlir_path: Path, config_name: str, compile_flags: list[str]):
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


def run_iree_module(iree_module_path: Path, run_flags: list[str]):
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


def compare_between_iree_and_onnxruntime_fn(model_url: str, artifacts_subdir=""):
    test_artifacts_dir = ARTIFACTS_ROOT / artifacts_subdir
    if not test_artifacts_dir.is_dir():
        test_artifacts_dir.mkdir(parents=True)

    # Extract path and file components from the model URL.
    # "https://github.com/.../mobilenetv2-12.onnx" --> "mobilenetv2-12.onnx"
    model_file_name = model_url.rsplit("/", 1)[-1]
    # "mobilenetv2-12.onnx" --> "mobilenetv2-12"
    model_name = model_file_name.rsplit(".", 1)[0]

    # Download the model as needed.
    # TODO(scotttodd): move to fixture with cache / download on demand
    # TODO(scotttodd): overwrite if already existing? check SHA?
    original_onnx_path = test_artifacts_dir / f"{model_name}.onnx"
    if not original_onnx_path.exists():
        urllib.request.urlretrieve(model_url, original_onnx_path)

    # TODO(scotttodd): cache ONNX metadata and runtime results (pickle?)
    upgraded_onnx_path = upgrade_onnx_model_version(original_onnx_path)

    onnx_model_metadata = get_onnx_model_metadata(upgraded_onnx_path)
    logger.debug("ONNX model metadata:")
    logger.debug(onnx_model_metadata)

    # Prepare inputs and expected outputs for running through IREE.
    run_module_args = []
    for input in onnx_model_metadata.inputs:
        run_module_args.append(f"--input={input.type}=@{input.data_file}")
    for output in onnx_model_metadata.outputs:
        run_module_args.append(f"--expected_output={output.type}=@{output.data_file}")

    # Import, compile, then run with IREE.
    imported_mlir_path = import_onnx_model_to_mlir(upgraded_onnx_path)
    iree_module_path = compile_mlir_with_iree(
        imported_mlir_path, "cpu", ["--iree-hal-target-backends=llvm-cpu"]
    )
    # Note: could load the output into memory here and compare using numpy
    # if the pass/fail criteria is difficult to model in the native tooling.
    run_flags = ["--device=local-task"]
    run_flags.extend(run_module_args)
    run_iree_module(iree_module_path, run_flags)


@pytest.fixture
def compare_between_iree_and_onnxruntime():
    return compare_between_iree_and_onnxruntime_fn
