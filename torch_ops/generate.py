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
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


import torch
import iree.turbine.aot as aot
from iree.turbine.kernel.boo.op_exports.conv import ConvSignature, Mode
from iree.turbine.kernel.boo.runtime import use_cache_dir

from common import CommonConfig, Formula, ensure_dir_exists, CustomJSONEncoder


def export(module, test_folder, file_name, export_kwargs, args_torch, kwargs_torch):
    if isinstance(module, ConvSignature):
        _module = module.get_compiled_module(backend="iree_boo_experimental")
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpfolder = Path(tmpdirname)
            with use_cache_dir(tmpfolder):
                _module(*args_torch, **kwargs_torch)
            entry_point_name = list(tmpfolder.glob("fused*"))
            assert len(entry_point_name) == 1, f"{entry_point_name}"
            export_kwargs["function_name"] = str(Path(entry_point_name[0]).name)
            mlir_files = list(tmpfolder.glob("**/*.mlir"))
            assert len(mlir_files) == 1, f"{mlir_files}"
            file = mlir_files[0]
            file.rename(test_folder / file_name)
            # Unlike the other case, we cannot set function name
            # ourselves so we return it.
            return export_kwargs["function_name"], file
    else:
        ensure_dir_exists(test_folder)
        exported_module = aot.export(module, **export_kwargs)
        output = test_folder / file_name
        exported_module.save_mlir(output)
        # We return function name to keep signature.
        return export_kwargs["function_name"], output


def save_expected_output(module, test_folder, args_torch, kwargs_torch):
    if isinstance(module, ConvSignature):
        func = module.get_compiled_module(backend="inductor")
    else:
        func = module.forward
    results = func(*args_torch, **kwargs_torch)
    results, _ = torch.utils._pytree.tree_flatten(results)
    expected_outputs = []
    for idx, result in enumerate(results):
        fname = f"expected_result_{idx}.npy"
        expected_outputs.append(fname)
        file = test_folder / fname
        np.save(file, result)
    return expected_outputs


def torch_dtype_to_numpy(dtype):
    match dtype:
        case torch.float16:
            return np.dtype("float16")
    raise ValueError(f"do not have equivalent type for {dtype}")


def formulas_to_torch(formulas):
    return [
        torch.from_numpy(formula.numpy()) if isinstance(formula, Formula) else formula
        for formula in formulas
    ]


def signature_to_formulas(
    signature: ConvSignature,
    *,
    device: str | torch.device | None = None,
    splat_value: int | float | None = None,
    seed: int | None = None,
):
    assert not device, "we do not expect a device"
    assert not seed, "we do not expect a seed"
    assert not splat_value, "we do not expect splat_value"

    def get(shape):
        return Formula(shape=shape, dtype=torch_dtype_to_numpy(signature.dtype))

    if signature.mode == Mode.FORWARD:
        # (x, w, b) or (x, w)
        return (
            (
                get(signature.input_shape),
                get(signature.kernel_shape),
                get(signature.out_channels),
            )
            if signature.bias
            else (get(signature.input_shape), get(signature.kernel_shape))
        )

    return (
        get(signature.output_shape),
        get(signature.input_shape),
        get(signature.kernel_shape),
    )


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
    _module = module
    if isinstance(module, ConvSignature):
        _module = module.get_nn_module()

    name = _module._get_name()
    config.class_dir = Path(config.path / name)
    config.test_dir = Path(config.class_dir / config.name)

    ensure_dir_exists(config.test_dir)
    config.function_name, mlir_file = export(
        module,
        config.test_dir,
        config.file_name,
        config.get_export_kwargs(),
        config.args_torch,
        config.kwargs_torch,
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
