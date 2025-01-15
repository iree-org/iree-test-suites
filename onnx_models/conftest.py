# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import logging
import os
import pyjson5
import pytest
import subprocess
import urllib.request
from dataclasses import dataclass
from onnxruntime import InferenceSession, SessionOptions
from pathlib import Path

from .utils import *

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent
ARTIFACTS_ROOT = THIS_DIR / "artifacts"


###############################################################################
# Configuration
###############################################################################


def pytest_addoption(parser):
    # List of configuration files following this schema:
    #   {
    #     "config_name": str,
    #     "iree_compile_flags": list of str,
    #     "iree_run_module_flags": list of str,
    #     "skip_compile_tests": list of str,
    #     "skip_run_tests": list of str,
    #     "tests_and_expected_outcomes": dict
    #   }
    #
    # For example, to run some tests on CPU with the `llvm-cpu` backend and
    # `local-task` device:
    #   {
    #     "config_name": "cpu_llvm_task",
    #     "iree_compile_flags": ["--iree-hal-target-backends=llvm-cpu"],
    #     "iree_run_module_flags": ["--device=local-task"],
    #     "tests_and_expected_outcomes": {
    #       "default": "skip",
    #       "tests/foo/bar/baz.py::test_a": "pass",
    #       "tests/foo/bar/baz.py::test_b[params/x]": "fail-import",
    #       "tests/foo/bar/baz.py::test_b[params/y]": "fail-import",
    #       "tests/foo/bar/baz.py::test_b[params/z]": "fail-import",
    #       "tests/foo/bar/baz.py::test_c": "fail-compile",
    #       "tests/foo/bar/baz.py::test_d": "fail-run"
    #     }
    #   }
    #
    # The file can be specified in (by order of preference):
    #   1. The `--config-file` argument
    #       e.g. `pytest ... --config-file foo.json`
    #   2. The `IREE_TEST_CONFIG_FILE` environment variable
    #       e.g. `export IREE_TEST_CONFIG_FILE=foo.json`
    #   3. A default config file used for testing the test suite itself
    default_config_file = os.getenv(
        "IREE_TEST_CONFIG_FILE", THIS_DIR / "configs" / "onnx_models_cpu_llvm_task.json"
    )
    parser.addoption(
        "--test-config-file",
        type=Path,
        default=default_config_file,
        help="Config JSON file used to parameterize test cases",
    )


def pytest_sessionstart(session):
    config_file_path = session.config.getoption("test_config_file")
    with open(config_file_path) as config_file:
        test_config = pyjson5.load(config_file)
    session.config.iree_test_config = test_config


def pytest_collection_modifyitems(session, config, items):
    logger.debug(f"pytest_collection_modifyitems with {len(items)} items:")

    tests_and_expected_outcomes = config.iree_test_config["tests_and_expected_outcomes"]
    default_outcome = tests_and_expected_outcomes.get("default", "skip")

    for item in items:
        # Build a test name from the test item location, matching how the test
        # appears in logs, e.g.
        # "tests/model_zoo/validated/vision/classification_models_test.py::test_alexnet"
        # https://docs.pytest.org/en/stable/reference/reference.html#pytest.Item
        standardized_location_0 = item.location[0].replace("\\", "/")
        item_path = f"{standardized_location_0}::{item.location[2]}"

        expected_outcome = tests_and_expected_outcomes.get(item_path, default_outcome)
        logger.debug(f"Expected outcome for {item_path} is {expected_outcome}")

        if expected_outcome == "skip":
            mark = pytest.mark.skip(reason="Test not included in config")
            item.add_marker(mark)
        elif expected_outcome == "pass":
            pass
        elif expected_outcome == "fail-import":
            mark = pytest.mark.xfail(raises=IreeImportOnnxException)
            item.add_marker(mark)
        elif expected_outcome == "fail-compile":
            mark = pytest.mark.xfail(raises=IreeCompileException)
            item.add_marker(mark)
        elif expected_outcome == "fail-run":
            mark = pytest.mark.xfail(raises=IreeRunException)
            item.add_marker(mark)


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


def get_onnx_model_metadata(onnx_path: Path) -> OnnxModelMetadata:
    # We can either
    #   A) List all metadata explicitly
    #   B) Get metadata on demand from the .onnx protobuf using 'onnx'
    #   C) Get metadata on demand from the InferenceSession using 'onnxruntime'
    # This is option C.

    so = SessionOptions()
    so.log_severity_level = 3  # ignore warnings
    onnx_session = InferenceSession(onnx_path, so)
    logger.info(f"Getting model metadata for '{onnx_path.relative_to(THIS_DIR)}'")
    inputs = []
    onnx_inputs = {}
    for idx, input in enumerate(onnx_session.get_inputs()):
        logger.debug(f"Session input [{idx}]")
        logger.debug(f"  name: '{input.name}'")
        iree_type = convert_ort_to_iree_type(input)
        logger.debug(f"  shape: {input.shape}")
        logger.debug(f"  type: '{input.type}'")
        logger.debug(f"  iree parameter: {iree_type}")

        # Create a numpy tensor with some random data for the input.
        input_data = generate_numpy_input_for_ort_node_arg(input)
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
        iree_type = convert_numpy_to_iree_type_string(result)
        logger.debug(f"Session output [{idx}]")
        logger.debug(f"  name: '{output.name}'")
        logger.debug(f"  shape (actual): {result.shape}")
        logger.debug(f"  type (numpy): '{result.dtype}'")
        logger.debug(f"  iree parameter: {iree_type}")
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


@pytest.fixture
def compare_between_iree_and_onnxruntime(pytestconfig):
    config_name = pytestconfig.iree_test_config["config_name"]
    iree_compile_flags = pytestconfig.iree_test_config["iree_compile_flags"]
    iree_run_module_flags = pytestconfig.iree_test_config["iree_run_module_flags"]

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
        # TODO(scotttodd): redownload if file is corrupted (e.g. partial download)
        onnx_path = test_artifacts_dir / f"{model_name}.onnx"
        if not onnx_path.exists():
            urllib.request.urlretrieve(model_url, onnx_path)

        # TODO(scotttodd): cache ONNX metadata and runtime results (pickle?)
        onnx_model_metadata = get_onnx_model_metadata(onnx_path)
        logger.debug(onnx_model_metadata)

        # Prepare inputs and expected outputs for running through IREE.
        run_module_args = []
        for input in onnx_model_metadata.inputs:
            run_module_args.append(
                f"--input={input.type}=@{input.data_file.relative_to(THIS_DIR)}"
            )
        for output in onnx_model_metadata.outputs:
            run_module_args.append(
                f"--expected_output={output.type}=@{output.data_file.relative_to(THIS_DIR)}"
            )

        # Import, compile, then run with IREE.
        imported_mlir_path = import_onnx_model_to_mlir(onnx_path)
        iree_module_path = compile_mlir_with_iree(
            imported_mlir_path, config_name, iree_compile_flags.copy()
        )
        # Note: could load the output into memory here and compare using numpy
        # if the pass/fail criteria is difficult to model in the native tooling.
        run_flags = iree_run_module_flags.copy()
        run_flags.extend(run_module_args)
        run_iree_module(iree_module_path, run_flags)

    return compare_between_iree_and_onnxruntime_fn
