# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import kagglehub
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent

###############################################################################
# Exception types
###############################################################################

# Note: can mark tests as expected to fail at a specific stage with:
# @pytest.mark.xfail(raises=KaggleHubDownloadException)
# @pytest.mark.xfail(raises=IreeImportTfLiteException)
# @pytest.mark.xfail(raises=IreeCompileException)
# @pytest.mark.xfail(raises=IreeRunException)


class KaggleHubDownloadException(RuntimeError):
    pass


class IreeImportTfLiteException(RuntimeError):
    pass


class IreeCompileException(RuntimeError):
    pass


###############################################################################
# LiteRT/TFLite loading, running, import, etc.
###############################################################################


def download_from_kagglehub(kaggle_model_name: str) -> Path:
    model_dir = Path(kagglehub.model_download(kaggle_model_name))
    tflite_paths = list(model_dir.glob("*.tflite"))
    if len(tflite_paths) != 1:
        raise KaggleHubDownloadException(
            f"Expected exactly one .tflite file in download directory: {model_dir}, found {tflite_paths}"
        )
    return tflite_paths[0]


def import_litert_model_to_mlir(model_path: Path) -> Path:
    # TODO(scotttodd): TEST_SUITE_ROOT/aritfacts dir like in onnx_models/

    imported_mlir_path = model_path.with_suffix(".mlirbc")
    logger.info(f"Importing '{model_path}' to '{imported_mlir_path}'")
    exec_args = [
        "iree-import-tflite",
        str(model_path),
        "-o",
        str(imported_mlir_path),
    ]
    ret = subprocess.run(exec_args, capture_output=True)
    if ret.returncode != 0:
        logger.error(f"Import of '{model_path.name}' failed!")
        logger.error("iree-import-tflite stdout:")
        logger.error(ret.stdout.decode("utf-8"))
        logger.error("iree-import-tflite stderr:")
        logger.error(ret.stderr.decode("utf-8"))
        raise IreeImportTfLiteException(f"  '{model_path.name}' import failed")
    return imported_mlir_path


###############################################################################
# IREE compilation and running
###############################################################################


def compile_mlir_with_iree(
    mlir_path: Path, config_name: str, compile_flags: list[str]
) -> Path:
    cwd = THIS_DIR
    iree_module_path = mlir_path.with_name(mlir_path.stem + f"_{config_name}.vmfb")
    compile_args = ["iree-compile", mlir_path]
    compile_args.extend(compile_flags)
    compile_args.extend(["-o", iree_module_path])
    compile_cmd = subprocess.list2cmdline(compile_args)
    logger.info(
        f"Launching compile command:\n"  #
        f"  cd {cwd} && {compile_cmd}"
    )
    ret = subprocess.run(compile_cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
        logger.error(f"Compilation of '{iree_module_path}' failed")
        logger.error("iree-compile stdout:")
        logger.error(ret.stdout.decode("utf-8"))
        logger.error("iree-compile stderr:")
        logger.error(ret.stderr.decode("utf-8"))
        raise IreeCompileException(f"  '{iree_module_path.name}' compile failed")
    return iree_module_path
