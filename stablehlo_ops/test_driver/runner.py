"""Compile source cases and compare against freshly evaluated references."""

from pathlib import Path
import shlex
import subprocess
from typing import Any

import numpy as np


class RunnerError(RuntimeError):
    """An IREE command failed or produced unexpected output."""


def compare_output(actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        raise RunnerError(
            f"Output shape mismatch: expected {expected.shape}, actual {actual.shape}"
        )
    matches = (
        np.array_equal(actual, expected)
        if expected.dtype.kind in "biu"
        else np.allclose(actual, expected, rtol=1e-5, atol=1e-8)
    )
    if not matches:
        raise RunnerError(f"Output mismatch:\nexpected: {expected}\nactual:   {actual}")


def run_command(command: list[str], timeout: float) -> None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        raise RunnerError(
            f"Command exceeded {timeout}s: {shlex.join(command)}"
        ) from error
    if result.returncode:
        raise RunnerError(
            f"Error invoking {Path(command[0]).name}\n"
            f"Command: {shlex.join(command)}\n{result.stdout}\n{result.stderr}"
        )


def compile_module(source: Path, config: dict[str, Any], vmfb: Path) -> None:
    compile_command = [
        "iree-compile",
        str(source),
        *config["iree_compile_flags"],
        "-o",
        str(vmfb),
    ]
    run_command(compile_command, config.get("compile_timeout", 120))


def save_iree_inputs(work: Path, inputs: list[np.ndarray]) -> list[Path]:
    paths = []
    for index, tensor in enumerate(inputs):
        path = work / f"input_{index}.npy"
        np.save(path, tensor, allow_pickle=False)
        paths.append(path)
    return paths


def run_module(
    input_files: list[Path],
    config: dict[str, Any],
    vmfb: Path,
    expected: list[np.ndarray],
) -> None:
    work = vmfb.parent
    outputs = [work / f"actual_{index}.npy" for index in range(len(expected))]
    command = [
        "iree-run-module",
        f"--module={vmfb}",
        "--function=main",
        *config["iree_run_module_flags"],
        *(f"--input=@{path}" for path in input_files),
        *(f"--output=@{output}" for output in outputs),
    ]
    run_command(command, config.get("run_timeout", 30))
    for index, (output, reference) in enumerate(zip(outputs, expected, strict=True)):
        if not output.is_file():
            raise RunnerError(f"IREE did not produce output {index}")
        actual = np.load(output, allow_pickle=False)
        compare_output(actual, reference)
