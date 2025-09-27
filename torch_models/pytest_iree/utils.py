from typing import Dict, Sequence, Union
from pathlib import Path
import subprocess
import time
import os
import json
import logging

logger = logging.getLogger(__name__)

IREE_COMPILE_QOL_FLAGS = [
    "--mlir-timing",
    "--mlir-timing-display=list",
    "--iree-consteval-jit-debug",
]


class IreeCompileException(RuntimeError):
    pass


class IreeRuntimeException(RuntimeError):
    pass


def iree_compile(source: Path, output: Path, cwd: Path, args: Sequence[str]):
    sep = "\n  "
    logger.info("**************************************************************")
    logger.info(f"  {sep.join(args)}")
    exec_args = (
        [
            "iree-compile",
            str(source.absolute()),
            "-o",
            str(output.absolute()),
        ]
        + IREE_COMPILE_QOL_FLAGS
        + list(args)
    )
    logger.info("Exec: " + " ".join(exec_args))
    start_time = time.time()
    ret = subprocess.run(exec_args, capture_output=True, cwd=cwd)
    if ret.returncode != 0:
        logger.error(f"Compilation of {str(source)} failed")
        logger.error("iree-compile stdout:")
        logger.error(ret.stdout.decode("utf-8"))
        logger.error("iree-compile stderr:")
        logger.error(ret.stderr.decode("utf-8"))
        raise IreeCompileException(f"Compilation of {str(source)} failed")
    run_time = time.time() - start_time
    logger.info(f"Compilation succeeded in {run_time}s")
    logger.info("**************************************************************")


def iree_run_module(modules: list[Path], cwd: Path, args: Sequence[str]):
    exec_args = [
        "iree-run-module",
    ] + [f"--module={str(m.absolute())}" for m in modules]
    exec_args.extend(args)
    logger.info("**************************************************************")
    logger.info("Exec: " + " ".join(exec_args))
    ret = subprocess.run(exec_args, capture_output=True, cwd=cwd)
    if ret.returncode != 0:
        logger.error(
            f"IREE run module of modules {', '.join([str(m) for m in modules])} failed"
        )
        logger.error("iree-run-module stdout:")
        logger.error(ret.stdout.decode("utf-8"))
        logger.error("iree-run-module stderr:")
        logger.error(ret.stderr.decode("utf-8"))
        raise IreeRuntimeException(
            f"IREE run module of modules {', '.join([str(m) for m in modules])} failed"
        )


def iree_benchmark_module(modules: list[Path], cwd: Path, args: Sequence[str]) -> dict:
    exec_args = [
        "iree-benchmark-module",
        "--benchmark_format=json",
    ] + [f"--module={str(m.absolute())}" for m in modules]
    exec_args.extend(args)
    logger.info("**************************************************************")
    logger.info("Exec: " + " ".join(exec_args))
    ret = subprocess.run(exec_args, capture_output=True, cwd=cwd)
    if ret.returncode != 0:
        logger.error(
            f"IREE run module of modules {', '.join([str(m) for m in modules])} failed"
        )
        logger.error("iree-run-module stdout:")
        logger.error(ret.stdout.decode("utf-8"))
        logger.error("iree-run-module stderr:")
        logger.error(ret.stderr.decode("utf-8"))
        raise IreeRuntimeException(
            f"IREE run module of modules {', '.join([str(m) for m in modules])} failed"
        )
    try:
        return json.loads(ret.stdout.decode("utf-8"))
    except json.JSONDecodeError as e:
        logger.error("Failed to parse benchmark output as JSON")
        logger.error(ret.stdout.decode("utf-8"))
        raise e
