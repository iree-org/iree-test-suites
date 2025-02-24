# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from typing import Dict, Sequence, Union
from pathlib import Path
import subprocess
import time
import os
import logging

logger = logging.getLogger(__name__)

from .artifacts import (
    Artifact,
    FetchedArtifact,
    ProducedArtifact,
)


IREE_COMPILE_QOL_FLAGS = [
    "--mlir-timing",
    "--mlir-timing-display=list",
    "--iree-consteval-jit-debug",
]


def fetch_source_fixture(url: str, *, group: str):
    art = FetchedArtifact(url=url, group=group)
    art.start()
    return art


def iree_compile(source: Artifact, flags: Sequence[str], vmfb_path: Path):
    if not os.path.exists(vmfb_path.parent):
        os.makedirs(vmfb_path.parent)
    sep = "\n  "
    logger.info("**************************************************************")
    logger.info(f"  {sep.join(flags)}")
    exec_args = (
        [
            "iree-compile",
            "-o",
            str(vmfb_path),
            str(source.path),
        ]
        + IREE_COMPILE_QOL_FLAGS
        + flags
    )
    logger.info("Exec: " + str(exec_args))
    start_time = time.time()
    subprocess.run(exec_args, check=True, capture_output=True, cwd=vmfb_path.parent)
    run_time = time.time() - start_time
    logger.info(f"Compilation succeeded in {run_time}s")
    logger.info("**************************************************************")
    return vmfb_path


def iree_run_module(vmfb: Path, *, device, function, args: Sequence[str] = ()):
    exec_args = [
        "iree-run-module",
        f"--device={device}",
        f"--module={vmfb}",
        f"--function={function}",
    ]
    exec_args.extend(args)
    logger.info("**************************************************************")
    logger.info("Exec: " + str(exec_args))
    subprocess.run(exec_args, check=True, capture_output=True, cwd=vmfb.parent)


def iree_benchmark_module(vmfb: Path, *, device, function, args: Sequence[str] = ()):
    exec_args = [
        "iree-benchmark-module",
        f"--device={device}",
        f"--function={function}",
    ]
    exec_args.extend(args)
    # if a pipeline module is being benchmarked, the module dependencies need to be ordered before the pipeline module
    exec_args.append(f"--module={vmfb}")
    logger.info("**************************************************************")
    logger.info("Exec: " + str(exec_args))
    proc = subprocess.run(
        exec_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        cwd=vmfb.parent,
    )
    return proc.returncode, proc.stdout
