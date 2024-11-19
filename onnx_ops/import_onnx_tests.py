# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import logging
import onnx
import shutil
import subprocess
from import_onnx_tests_utils import *
from multiprocessing import Pool
from pathlib import Path

logger = logging.getLogger(__name__)

ONNX_PACKAGE_DIR = Path(onnx.__file__).parent
ONNX_NODE_TESTS_ROOT = ONNX_PACKAGE_DIR / "backend/test/data/node"

# Convert test cases to at least this version using The ONNX Version Converter.
ONNX_CONVERTER_OUTPUT_MIN_VERSION = 17

# Write generated files to a subfolder.
THIS_DIR = Path(__file__).parent
GENERATED_FILES_OUTPUT_ROOT = THIS_DIR / "onnx/node/generated"
IMPORT_SUCCESSES_FILE_PATH = GENERATED_FILES_OUTPUT_ROOT / "import_successes.txt"
IMPORT_FAILURES_FILE_PATH = GENERATED_FILES_OUTPUT_ROOT / "import_failures.txt"


def find_onnx_tests(root_dir_path: Path):
    test_dir_paths = [p for p in root_dir_path.iterdir() if p.is_dir()]
    print(f"Found {len(test_dir_paths)} tests under '{root_dir_path}'")
    return sorted(test_dir_paths)


def import_onnx_files_with_cleanup(test_dir_path: Path):
    test_name = test_dir_path.name
    imported_dir_path = Path(GENERATED_FILES_OUTPUT_ROOT) / test_name
    try:
        import_onnx_files(test_dir_path, imported_dir_path)
    except Exception as e:
        logger.error(f"Error importing {test_name}: {e}")
        # Note: could comment this out to keep partially imported directories.
        shutil.rmtree(imported_dir_path)
        return (test_name, False)
    return (test_name, True)


def import_onnx_files(test_dir_path: Path, imported_dir_path: Path):
    # This imports one 'test_[name]' subfolder from this:
    #
    #   test_[name]/
    #     model.onnx
    #     test_data_set_0/
    #       input_0.pb
    #       output_0.pb
    #
    # to this:
    #
    #   imported_dir_path/...
    #     test_[name]/
    #       model.mlir  (torch-mlir)
    #       input_0.npy
    #       output_0.npy
    #       run_module_io_flags.txt  (flagfile with --input=input_0.npy, --expected_output=)

    imported_dir_path.mkdir(parents=True, exist_ok=True)

    test_data_flagfile_path = imported_dir_path / "run_module_io_flags.txt"
    test_data_flagfile_lines = []

    # Import model.onnx to model.mlir.
    model_path = test_dir_path / "model.onnx"
    imported_model_path = imported_dir_path / "model.mlir"
    exec_args = [
        "iree-import-onnx",
        str(model_path),
        "--opset-version",
        str(ONNX_CONVERTER_OUTPUT_MIN_VERSION),
        "-o",
        str(imported_model_path),
    ]
    ret = subprocess.run(exec_args, capture_output=True)
    if ret.returncode != 0:
        logger.error(
            f"Import of {imported_dir_path.name} failed!\niree-import-onnx stdout:"
        )
        logger.error(ret.stdout.decode("utf-8"))
        logger.error("iree-import-onnx stderr:")
        logger.error(ret.stderr.decode("utf-8"))
        raise RuntimeError(f"  {imported_dir_path.name} import failed")

    test_data_dirs = sorted(test_dir_path.glob("test_data_set*"))
    if len(test_data_dirs) != 1:
        raise NotImplementedError(
            f"Unhandled 'len(test_data_dirs) != 1' for {imported_dir_path.name}"
        )

    # Convert from:
    #   * input/output_*.pb
    # to:
    #   * input/output_*.bin (little endian binary files)
    #   * flagfile.txt with
    #     `--input={SHAPE}x{DTYPE}=@input_*.bin` and
    #     `--expected_output={SHAPE}x{DTYPE}=@output_*.bin`
    test_data_dir = test_data_dirs[0]
    test_inputs = sorted(list(test_data_dir.glob("input_*.pb")))
    test_outputs = sorted(list(test_data_dir.glob("output_*.pb")))
    model = onnx.load(model_path)
    for i in range(len(test_inputs)):
        test_input = test_inputs[i]
        type_proto = model.graph.input[i].type
        converted_data = convert_onnx_proto_to_numpy_array(test_input, type_proto)
        converted_type = convert_onnx_type_proto_to_iree_type_string(type_proto)

        input_path_bin = (imported_dir_path / test_input.stem).with_suffix(".bin")
        write_binary_to_file(converted_data, input_path_bin)
        test_data_flagfile_lines.append(
            f"--input={converted_type}=@{input_path_bin.name}\n"
        )

    for i in range(len(test_outputs)):
        test_output = test_outputs[i]
        type_proto = model.graph.output[i].type
        converted_data = convert_onnx_proto_to_numpy_array(test_output, type_proto)
        converted_type = convert_onnx_type_proto_to_iree_type_string(type_proto)

        output_path_bin = (imported_dir_path / test_output.stem).with_suffix(".bin")
        write_binary_to_file(converted_data, output_path_bin)
        test_data_flagfile_lines.append(
            f"--expected_output={converted_type}=@{output_path_bin.name}\n"
        )

    with open(test_data_flagfile_path, "wt") as f:
        f.writelines(test_data_flagfile_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX test case importer.")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=8,
        help="Number of parallel processes to use when importing test cases",
    )
    args = parser.parse_args()

    if not GENERATED_FILES_OUTPUT_ROOT.is_dir():
        GENERATED_FILES_OUTPUT_ROOT.mkdir(parents=True)
    # TODO(scotttodd): add flag to not clear output dir?
    print(f"Clearing old generated files from '{GENERATED_FILES_OUTPUT_ROOT}'")
    shutil.rmtree(GENERATED_FILES_OUTPUT_ROOT)

    test_dir_paths = find_onnx_tests(ONNX_NODE_TESTS_ROOT)
    print(f"Importing tests in '{ONNX_NODE_TESTS_ROOT}'")
    print("******************************************************************")
    passed_imports = []
    failed_imports = []
    with Pool(args.jobs) as pool:
        results = pool.imap_unordered(import_onnx_files_with_cleanup, test_dir_paths)
        for result in results:
            if result[1]:
                passed_imports.append(result[0])
            else:
                failed_imports.append(result[0])
    print("******************************************************************")

    passed_imports.sort()
    failed_imports.sort()

    with open(IMPORT_SUCCESSES_FILE_PATH, "wt") as f:
        f.write("\n".join(passed_imports))
    with open(IMPORT_FAILURES_FILE_PATH, "wt") as f:
        f.write("\n".join(failed_imports))

    print(f"Import pass count: {len(passed_imports)}")
    print(f"Import fail count: {len(failed_imports)}")
