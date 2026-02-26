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
import pwd
import shutil
import tempfile
from typing import Any
import ml_dtypes


import torch
import iree.turbine.aot as aot
from iree.turbine.kernel.boo.op_exports.conv import ConvSignature, Mode
from iree.turbine.kernel.boo.runtime import use_cache_dir

from common import ArgSpec, CommonConfig, Formula, ensure_dir_exists, CustomJSONEncoder


def export(module, test_folder, file_name, export_kwargs, args_torch, kwargs_torch):
    if isinstance(module, ConvSignature):
        _module = module.get_compiled_module(backend="iree_boo_experimental")
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpfolder = Path(tmpdirname)
            with use_cache_dir(tmpfolder):
                _module(*args_torch, **kwargs_torch)
            entry_point_name = list(tmpfolder.glob("*"))
            assert len(entry_point_name) == 1, f"{entry_point_name}"
            function_name = str(Path(entry_point_name[0]).name)
            mlir_files = list(tmpfolder.glob("**/*.mlir"))
            assert len(mlir_files) == 1, f"{mlir_files}"
            file = mlir_files[0]
            shutil.move(str(file), test_folder / file_name)
            # Unlike the other case, we cannot set function name
            # ourselves so we return it.
            return function_name, file
    else:
        ensure_dir_exists(test_folder)
        exported_module = aot.export(module, **export_kwargs)
        output = test_folder / file_name
        exported_module.save_mlir(output)
        # We return function name to keep signature.
        return export_kwargs["function_name"], output


ACCOUNT = "sharkpublic"
CONTAINER = "sharkpublic"


def save_expected_output(
    module, test_folder, args_torch, kwargs_torch, prepare_azure_script=False
):
    if isinstance(module, ConvSignature):
        func = module.get_nn_module()
    else:
        func = module.forward
    results = func(*args_torch, **kwargs_torch)
    results, _ = torch.utils._pytree.tree_flatten(results)
    expected_outputs = []
    for idx, result in enumerate(results):
        fname = f"expected_result_{idx}.npy"
        file = test_folder / fname
        if result.dtype == torch.bfloat16:
            # torch's .numpy() doesn't support bfloat16, so round-trip through
            # uint16 views and use ml_dtypes for the numpy dtype.
            import ml_dtypes

            result_np = result.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
        else:
            result_np = result.detach().numpy()
        np.save(file, result_np)

        if prepare_azure_script:
            account_name = f"--account-name {ACCOUNT}"
            container_name = f"--container-name {CONTAINER}"
            uid = os.getuid()
            username = pwd.getpwuid(uid).pw_name
            name = Path(username) / Path(*file.parts[1:])
            cmd = f"az storage blob upload {account_name} {container_name} --name {name} --file {file}\n"
            with open("upload.sh", "a") as f:
                f.write(cmd)

            url = f"https://{ACCOUNT}.blob.core.windows.net/{CONTAINER}/{name}"
            expected_outputs.append(ArgSpec(url))
        else:
            expected_outputs.append(ArgSpec(value=fname))

    return expected_outputs


def torch_dtype_to_numpy(dtype):
    match dtype:
        case torch.float16:
            return np.dtype("float16")
        case torch.bfloat16:
            return np.dtype(ml_dtypes.bfloat16)
    raise ValueError(f"do not have equivalent type for {dtype}")


def formulas_to_torch(formulas):
    result: list[torch.Tensor] = []
    for formula in formulas:
        if not isinstance(formula, Formula):
            result.append(formula)
            continue
        arr = formula.numpy()
        if arr.dtype == ml_dtypes.bfloat16:
            # torch.from_numpy() doesn't support ml_dtypes.bfloat16, so view as uint16.
            t = torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
        else:
            t = torch.from_numpy(arr)
        result.append(t)
    return result


def signature_to_formulas(
    signature: ConvSignature,
    *,
    device: str | torch.device | None = None,
    splat_value: int | float | None = None,
    seed: int | None = None,
):
    """
    This function replaces get_sample_args.

    Unlike get_sample_args, we do not care about setting a device here, nor a seed.
    The seed is generated on a test level for the test-suite.

    The reason why we assert on the splat_value is that that is a level of control
    that does not yet exist in formula and we should honour it if possible.
    """
    assert not device, "we do not expect a device"
    assert not seed, "we do not expect a seed"
    assert not splat_value, "we do not expect splat_value"

    # Even if this function is intended to replace get_sample_args,
    # we use get_sample_args here to extract the shapes + dtypes.
    # That should be more robust to changes, and should extend easier to non-conv ops in the future.
    sample_args = signature.get_sample_args()
    formula_args = []
    for arg in sample_args:
        assert arg.is_contiguous(), "we expect arguments to be contiguous"
        dtype = torch_dtype_to_numpy(arg.dtype)
        formula_args.append(Formula(shape=arg.shape, dtype=dtype))

    return formula_args


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
    azure_blob: bool = True
    """Whether or not to store as an azure blob"""

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
            module,
            config.test_dir,
            config.args_torch,
            config.kwargs_torch,
            config.azure_blob,
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
