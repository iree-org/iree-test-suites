"""
# Summary

This file contains the necessary classes to generate differential
tests and performance regression tests. This file does not run the
tests, but just generates them. One of the reasons for this split
is to remove torch as a dependency when running the tests.

Each generated test corresponds to one directory which may contain:
    - run_module_io.json
    - test.mlir
    - expected_output_${idx}.npy files

However, when running the tests, it is also necessary to point
to a target configuration file (usually under configs) which
primarily provides information about which target to use when
compiling and other flags used when running.

TODO: Add scripts for running tests from the command line and
document them here and setting golden time.

The differential tests verify that given the same input,
a torch program will output the same result when using torch's
default backend and using IREE.

The performance regression tests verify that we do not exceed
a golden time.
"""

from pathlib import Path
from dataclasses import dataclass
import json
import numbers
import numpy as np
import subprocess
from typing import Any

# Import optional libraries.
# These libraries are only needed when
# the tests are generated. Not when the
# tests are run.
try:
    import torch
except:
    ...

try:
    import iree.turbine.aot as aot
except:
    ...


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


def export(module, test_folder, file_name, export_kwargs):
    ensure_dir_exists(test_folder)
    exported_module = aot.export(module, **export_kwargs)
    output = test_folder / file_name
    exported_module.save_mlir(output)
    return output


def formulas_to_torch(formulas):
    return [
        torch.from_numpy(formula.numpy()) if isinstance(formula, Formula) else formula
        for formula in formulas
    ]


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


def save_expected_output(module, test_folder, args_torch, kwargs_torch):
    results = module.forward(*args_torch, **kwargs_torch)
    results, _ = torch.utils._pytree.tree_flatten(results)
    expected_outputs = []
    for idx, result in enumerate(results):
        fname = f"expected_result_{idx}.npy"
        expected_outputs.append(fname)
        file = test_folder / fname
        np.save(file, result)
    return expected_outputs


@dataclass
class GenConfig:
    """
    Class holding parameters sent to aot.export.
    The one difference is that aot.export expects
    torch.Tensor arguments while this one expects
    them to be in Formula
    """

    name: str
    """Name of the test. Will be name of directory containing test files."""

    # Same as aot.export's parameters
    # These are used when generating the MLIR file.
    # function_name is also needed when running the test.
    args: tuple[Formula] | None = None
    """Arguments (as formulas instead of torch.Tensor)."""
    kwargs: dict[Any, Any] | None = None
    """Kwargs (as formulas instead of torch.Tensor)."""
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None
    """Dynamic shapes spec."""
    module_name: str | None = None
    """Name of module, will also be used as name of folder."""
    function_name: str | None = "main"
    """Identifier given to the entry_point of the module."""
    strict_export: bool = True
    """Strict export."""
    import_symbolic_shape_expressions: bool = False
    """Import symbolic shape expressions."""
    arg_device: dict[int, Any] | None = None
    """Arg device"""

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

    # These will definitely not be stored in run_module_io.json.
    # These are just here for convienience.
    args_torch: list["torch.Tensor"] | None = None
    """Formulas are now torch.Tensors."""
    kwargs_torch: dict[Any, Any] | None = None
    """Formulas are now torch.Tensors."""

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
            return GenConfig(
                name=test_dir.name,
                path=path,
                class_dir=class_dir,
                test_dir=test_dir,
                **json.load(f, object_hook=customJSONDecoder),
            )

    def flatten(self):
        args = self.args
        kwargs = self.kwargs
        args, _ = torch.utils._pytree.tree_flatten(args)
        kwargs, _ = torch.utils._pytree.tree_flatten(kwargs)
        return args + kwargs

    def to_torch(self, seed):
        """Only args and kwargs may hold Formulas.

        Since args and kwargs may be arbitrary Python data structures,
        use pytrees to flatten them and change Formula to torch.Tensors.
        """
        self.args = self.args if self.args is not None else ()
        self.kwargs = self.kwargs if self.kwargs is not None else {}

        args = self.args
        kwargs = self.kwargs

        args, args_shape = torch.utils._pytree.tree_flatten(args)
        kwargs, kwargs_shape = torch.utils._pytree.tree_flatten(kwargs)
        np.random.seed(seed)
        args = formulas_to_torch(args)
        kwargs = formulas_to_torch(kwargs)
        args = torch.utils._pytree.tree_unflatten(args, args_shape)
        kwargs = torch.utils._pytree.tree_unflatten(kwargs, kwargs_shape)

        self.args_torch = tuple(args)
        self.kwargs_torch = kwargs
        return self.args_torch, self.kwargs_torch

    def get_export_kwargs(self):
        self.to_torch(self.seed)
        return {
            "args": self.args_torch,
            "kwargs": self.kwargs_torch,
            "dynamic_shapes": self.dynamic_shapes,
            "module_name": self.module_name,
            "function_name": self.function_name,
            "strict_export": self.strict_export,
            "import_symbolic_shape_expressions": self.import_symbolic_shape_expressions,
            "arg_device": self.arg_device,
        }

    def get_run_module_io(self):
        flat_args = self.flat_args if self.flat_args is not None else self.flatten()
        data = {
            "function_name": self.function_name,
            "flat_args": flat_args,
            "seed": self.seed,
            "rtol": self.rtol,
            "atol": self.atol,
            "equal_nan": self.equal_nan,
            "mode": self.mode,
            "file_name": self.file_name,
            "vmfb_name": self.vmfb_name,
        }
        if self.mode == "compare":
            data["expected_output"] = self.expected_output
        return data

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

    def save_config(self):
        data = self.get_run_module_io()
        with open(self.test_dir / "run_module_io.json", "w") as file:
            json.dump(data, file, indent=4, cls=CustomJSONEncoder)
            print("", file=file)

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
            input_mlir_file = self.test_file / self.file_name
            raise IreeRunException(
                self.test_dir, proc, input_mlir_file, self.compile_cmd, run_cmd
            )

        self.compare_results(self.expected_output, output_files)

    def iree_benchmark_module(
        self, iree_run_flags, golden_time=float("nan"), return_golden_time=True
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
        real_time = output["benchmarks"][0]["real_time"]
        if return_golden_time:
            return real_time
        assert real_time <= golden_time

    def report_golden_time(self, iree_compile_flags, iree_run_flags):
        self.iree_compile(iree_compile_flags)
        golden_time = self.iree_benchmark_module(
            iree_run_flags, return_golden_time=True
        )
        print(golden_time * 1.1)
        return golden_time * 1.1

    def run_benchmark_test(
        self,
        iree_compile_flags,
        iree_run_flags,
        golden_time=float("nan"),
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

        if np.isnan(golden_time):
            raise ValueError("golden time has not been set!")
        self.iree_compile(iree_compile_flags)
        if skip_run:
            return
        self.iree_benchmark_module(iree_run_flags, golden_time)

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


def gen(module, configs):
    """Generate test with multiple configurations."""
    for config in configs:
        gen_config(module, config)


def gen_config(module, config):
    """Generate test with single configuration."""
    config.class_dir = Path(config.path / module._get_name())
    config.test_dir = Path(config.class_dir / config.name)

    ensure_dir_exists(config.test_dir)
    mlir_file = export(
        module, config.test_dir, config.file_name, config.get_export_kwargs()
    )
    # TODO: inline this inside export. Make export a method of GenConfig
    if config.mode == "compare":
        config.expected_output = save_expected_output(
            module, config.test_dir, config.args_torch, config.kwargs_torch
        )
    config.save_config()


def test(function):
    """Mark function as test."""
    function._is_test = True
    return function


class ModuleWrapper:
    """Wrapper around torch.nn.Module

    If called, it will call the module's constructor
    and generate the tests for the specific module.
    Tests data comes from functions mark with the _is_test attribute.
    """

    def __init__(self, cls):
        self.cls = cls

    def __call__(self, *args, **kwargs):
        module = self.cls(*args, **kwargs)
        for attr in dir(module):
            attr = getattr(module, attr)
            if not hasattr(attr, "_is_test"):
                continue

            configs = attr
            gen(module, configs())


def gen_tests(cls):
    """Decorator for torch.nn.Modules"""
    return ModuleWrapper(cls)
