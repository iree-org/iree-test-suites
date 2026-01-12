"""
# Summary

File containing the necessary classes and methods for generating tests.

Each generated test corresponds to one directory which may contain:
    - run_module_io.json
    - test.mlir
    - expected_output_${idx}.npy files

TODO: Add scripts for running tests from the command line and
document them here and setting golden time.

The differential tests verify that given the same input,
a torch program will output the same result when using torch's
default backend and using IREE.

The performance regression tests verify that we do not exceed
a golden time.

"""
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path
from typing import Any


import torch
import iree.turbine.aot as aot

from common import CommonConfig, Formula, ensure_dir_exists, CustomJSONEncoder


def export(module, test_folder, file_name, export_kwargs):
    ensure_dir_exists(test_folder)
    exported_module = aot.export(module, **export_kwargs)
    output = test_folder / file_name
    exported_module.save_mlir(output)
    return output


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


def formulas_to_torch(formulas):
    return [
        torch.from_numpy(formula.numpy()) if isinstance(formula, Formula) else formula
        for formula in formulas
    ]


@dataclass
class GenConfig(CommonConfig):
    """Class used during test generation"""

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
    # There's also function_name
    # But that is on CommonConfig
    strict_export: bool = True
    """Strict export."""
    import_symbolic_shape_expressions: bool = False
    """Import symbolic shape expressions."""
    arg_device: dict[int, Any] | None = None
    """Arg device"""

    # These will definitely not be stored in run_module_io.json.
    # These are just here for convienience.
    args_torch: list["torch.Tensor"] | None = None
    """Formulas are now torch.Tensors."""
    kwargs_torch: dict[Any, Any] | None = None
    """Formulas are now torch.Tensors."""

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

    def flatten(self):
        args = self.args
        kwargs = self.kwargs
        args, _ = torch.utils._pytree.tree_flatten(args)
        return args

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

    def save_config(self):
        data = self.get_run_module_io()
        with open(self.test_dir / "run_module_io.json", "w") as file:
            json.dump(data, file, indent=4, cls=CustomJSONEncoder)
            print("", file=file)

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
