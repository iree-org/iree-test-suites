"""
# Summary

This file contains the classes and methods common to generating and
running tests. This file does not run the tests nor generates them.
It does contain the logic for running the tests, but to run them
one needs to use pytest.
For generation, look at the methods and classes in generate.py.

The generate.py file is separate from this common file to make it explicit
where torch is used as a dependency. Here, torch should never be imported.

When running the tests, it is also necessary to point
to a target configuration file (usually under configs) which
primarily provides information about which target to use when
compiling and other flags used when running.
"""

from pathlib import Path
from dataclasses import dataclass
import json
import numbers
import numpy as np
import subprocess
from typing import Any

THIS_DIR = Path(__file__).parent


class IreeCompileException(Exception):
    """Compiler exception that preserves the command line and output."""

    def __init__(
        self,
        process: subprocess.CompletedProcess,
        cwd: Path,
        input_mlir_file: Path,
        compile_cmd: str,
    ):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)
        try:
            outs = process.stdout.decode("utf-8")
        except:
            outs = str(process.stdout)

        test_github_url = (
            "https://github.com/iree-org/iree-test-suites/blob/main/torch_ops/"
            + cwd.relative_to(THIS_DIR).as_posix()
        )

        with open(input_mlir_file) as f:
            input_mlir = f.read()

            super().__init__(
                f"Error invoking iree-compile\n"
                f"Error code: {process.returncode}\n"
                f"Stderr diagnostics:\n{errs}\n\n"
                f"Stdout diagnostics:\n{outs}\n\n"
                f"Test case source:\n"
                f"  {test_github_url}\n\n"
                f"Input program:\n"
                f"```\n{input_mlir}```\n\n"
                f"Compiled with:\n"
                f"  cd {cwd} && {compile_cmd}\n\n"
            )


class IreeRunException(Exception):
    """Runtime exception that preserves the command line and output."""

    def __init__(
        self,
        cwd: Path,
        process: subprocess.CompletedProcess,
        input_mlir_file: Path,
        compile_cmd: str,
        run_cmd: str,
    ):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)
        try:
            outs = process.stdout.decode("utf-8")
        except:
            outs = str(process.stdout)

        test_github_url = (
            "https://github.com/iree-org/iree-test-suites/blob/main/torch_ops/"
            + cwd.relative_to(THIS_DIR).as_posix()
        )

        with open(input_mlir_file) as f:
            input_mlir = f.read()

            super().__init__(
                f"Error invoking iree-run-module\n"
                f"Error code: {process.returncode}\n"
                f"Stderr diagnostics:\n{errs}\n"
                f"Stdout diagnostics:\n{outs}\n"
                f"Test case source:\n"
                f"  {test_github_url}\n\n"
                f"Input program:\n"
                f"```\n{input_mlir}```\n\n"
                f"Compiled with:\n"
                f"  cd {cwd} && {compile_cmd}\n\n"
                f"Run with:\n"
                f"  cd {cwd} && {run_cmd}\n\n"
            )


class IreeXFailCompileRunException(Exception):
    pass


@dataclass(frozen=True, kw_only=True)
class Formula:
    """
    Represents the formula:

    (coeff * numpy.random.rand(*shape) + offset).astype(dtype)

    This class is useful because it allows us to compress the input tensors.
    Without this class, one would need to store all inputs to all programs.
    When dealing with a small set of tests, this may be ok. But when dealing
    with larger sets of tests with large tensors, storing all of these files
    with random tensor data does not make a lot of sense.

    With this class, benchmark tests (which do not need to compare the
    output) do not store tensors.

    We still control and guarantee that the same input is generated for every test
    by setting the seed before generating the contents of each tensor produced by
    Formula. The seed is set on a per-test basis (as opposed to on a per-Formula
    basis).
    """

    shape: tuple[int, ...]
    dtype: type = np.dtype("float32")
    coeff: numbers.Number = 1
    offset: numbers.Number = 0

    def numpy(self):
        return (self.coeff * np.random.rand(*self.shape) + self.offset).astype(
            self.dtype
        )

    def toJSONEncoder(self):
        """
        Ensure all fields in dataclass are able to be encoded into JSON.

        self.dtype:type cannot be encoded into JSON
        """
        return {
            "Formula": {
                "shape": self.shape,
                "dtype": self.dtype.name,
                "coeff": self.coeff,
                "offset": self.offset,
            }
        }


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Formula):
            return obj.toJSONEncoder()
        return super().default(obj)


def customJSONDecoder(d):
    if kwargs := d.get("Formula"):
        kwargs["dtype"] = np.dtype(kwargs["dtype"])
        return Formula(**kwargs)
    return d


def ensure_dir_exists(path):
    path.mkdir(exist_ok=True, parents=True)
    return path


def formulas_to_npy(formulas):
    return [
        formula.numpy() if isinstance(formula, Formula) else formula
        for formula in formulas
    ]


def formulas_to_npy_files(formulas, test_folder, seed):
    np.random.seed(seed)
    tensors = formulas_to_npy(formulas)
    inputs = []
    for idx, tensor in enumerate(tensors):
        fname = f"input_{idx}.npy"
        inputs.append(fname)
        file = test_folder / fname
        np.save(file, tensor)
    return inputs


@dataclass
class CommonConfig:
    """Class used for running tests.

    Some of these fields may also be used during test generation.
    """

    name: str
    """Name of the test. Will be name of directory containing test files."""

    function_name: str | None = "main"
    """Identifier given to the entry_point of the module."""

    # Extra parameters.
    # These are used when compiling and running tests.
    file_name: str = "test.mlir"
    """Name of mlir file."""
    vmfb_name: str = "out.vmfb"
    """Name of vmfb file (used when compiling)."""
    seed: int = 0
    """Seed to be used for the generation of tensors."""
    rtol: float = 1e-05
    """The relative tolerance."""
    atol: float = 1e-08
    """The absolute tolerance."""
    equal_nan: bool = False
    """Flat list of arguments to be used during test"""
    mode: str = "compare"
    """Test mode."""

    # Directory hierarchy.
    path: Path = Path("generated")
    """Path to folder which will contain all generated tests."""

    # You do not need to set any of the ones below. They will be
    # set automatically.
    class_dir: Path | None = None
    """Class directory. Generated from module"""
    test_dir: Path | None = None
    """Test directory. Generated from test name"""
    expected_output: list[str] | None = None
    """List of npy files with expected outputs."""
    flat_args: list[Formula] | None = None
    """Flattened args and kwargs. Used when using iree-run-module."""

    # We just need to store this one when running tests
    # so that we can have nice run exceptions which include
    # the command used to compile the mlir file.
    compile_cmd: str | None = None
    """Compile command."""

    @property
    def qualified_name(self):
        return f"{self.class_dir.name}/{self.test_dir.name}"

    @staticmethod
    def load(run_module_io_json):
        test_dir = run_module_io_json.parent
        class_dir = test_dir.parent
        path = class_dir.parent
        name = test_dir.name
        with open(run_module_io_json, "r") as f:
            return CommonConfig(
                name=test_dir.name,
                path=path,
                class_dir=class_dir,
                test_dir=test_dir,
                **json.load(f, object_hook=customJSONDecoder),
            )

    def get_input_flags(self):
        args = self.flat_args
        input_npy_files = formulas_to_npy_files(args, self.test_dir, self.seed)

        flags = []
        for file in input_npy_files:
            path = self.test_dir / file
            option = f"--input=@{path}"
            flags.append(option)
        return flags

    def get_output_flags(self):
        files = []
        flags = []
        for idx, _ in enumerate(self.expected_output):
            fname = f"expected_output_{idx}.npy"
            files.append(fname)
            path = self.test_dir / fname
            option = f"--output=@{path}"
            flags.append(option)
        return files, flags

    def iree_compile(self, iree_compile_flags):
        mlir_file = self.test_dir / self.file_name
        vmfb_file = self.test_dir / self.vmfb_name
        self.compile_cmd = subprocess.list2cmdline(
            ["iree-compile", *iree_compile_flags, mlir_file, "-o", vmfb_file]
        )
        proc = subprocess.run(self.compile_cmd, shell=True, capture_output=True)
        if proc.returncode != 0:
            raise IreeCompileException(proc, self.test_dir, mlir_file, self.compile_cmd)

    def compare_results(self, expected, observed):
        rtol = self.rtol
        atol = self.atol
        equal_nan = self.equal_nan
        for exp, obs in zip(expected, observed, strict=True):
            obs_tensor = np.load(self.test_dir / obs)
            exp_tensor = np.load(self.test_dir / exp)
            assert np.allclose(
                obs_tensor, exp_tensor, rtol=rtol, atol=atol, equal_nan=equal_nan
            )

    def iree_run_module(self, iree_run_flags):
        module = self.test_dir / self.vmfb_name
        function = self.function_name
        input_flags = self.get_input_flags()
        output_files, output_flags = self.get_output_flags()
        run_cmd = [
            "iree-run-module",
            f"--module={module}",
            f"--function={function}",
            *iree_run_flags,
            *input_flags,
            *output_flags,
        ]
        run_cmd = subprocess.list2cmdline(run_cmd)
        proc = subprocess.run(run_cmd, shell=True, capture_output=True)
        if proc.returncode != 0:
            input_mlir_file = self.test_dir / self.file_name
            raise IreeRunException(
                self.test_dir, proc, input_mlir_file, self.compile_cmd, run_cmd
            )

        self.compare_results(self.expected_output, output_files)

    def iree_benchmark_module(
        self, iree_run_flags, golden_time_ms=float("nan"), return_golden_time=True
    ):
        module = self.test_dir / self.vmfb_name
        function = self.function_name
        input_flags = self.get_input_flags()
        cmd = [
            "iree-benchmark-module",
            f"--module={module}",
            f"--function={function}",
            "--benchmark_format=json",
            *iree_run_flags,
            *input_flags,
        ]
        cmd = subprocess.list2cmdline(cmd)
        proc = subprocess.run(cmd, shell=True, capture_output=True)
        if proc.returncode != 0:
            input_mlir_file = self.test_file / self.file_name
            raise IreeRunException(
                self.test_dir, proc, input_mlir_file, self.compile_cmd, run_cmd
            )

        output = json.loads(proc.stdout.decode("utf-8"))
        real_time_ms = output["benchmarks"][0]["real_time"]
        if return_golden_time:
            return real_time_ms
        assert real_time_ms <= golden_time_ms

    def report_golden_time(self, iree_compile_flags, iree_run_flags):
        self.iree_compile(iree_compile_flags)
        golden_time = self.iree_benchmark_module(
            iree_run_flags, return_golden_time=True
        )
        # 10% is added for variance.
        return golden_time * 1.1

    def run_benchmark_test(
        self,
        iree_compile_flags,
        iree_run_flags,
        golden_time_ms=float("nan"),
        skip_run=False,
    ):
        """Run benchmark test
        iree-compile \
                ${self.file_name} \
                ${iree_compile_flags} \
                -o ${self.vmfb_name}

        iree-benchmark-module \
                --benchmark_format=json \
                --module ${self.vmfb_name} \
                --function ${self.function_name} \
                --input=${self.flat_arg[0]} ... \
                --output=${output_file}

        assert real_time <= golden_time
        """

        report_time = np.isnan(golden_time_ms) and not skip_run
        if report_time:
            new_golden_time_ms = self.report_golden_time(
                iree_compile_flags, iree_run_flags
            )
            raise ValueError(
                f"golden time has not been set. Set to: {new_golden_time_ms} ms")

        self.iree_compile(iree_compile_flags)
        if skip_run:
            return

        self.iree_benchmark_module(iree_run_flags, golden_time_ms)

    def run_quality_test(self, iree_compile_flags, iree_run_flags, skip_run=False):
        """Run differential test.

        iree-compile \
                ${self.file_name} \
                ${iree_compile_flags} \
                -o ${self.vmfb_name}

        iree-run-module \
                --module ${self.vmfb_name} \
                --function ${self.function_name} \
                --input=${self.flat_arg[0]} ... \
                --output=${output_file}

        assert numpy.allclose(expected, observed, rtol=${self.rtol}, atol=${self.atol}, equal_nan=${self.equal_nan})
        """
        self.iree_compile(iree_compile_flags)
        if skip_run:
            return
        self.iree_run_module(iree_run_flags)
