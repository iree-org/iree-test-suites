import dataclasses
import functools
import json
import numbers
import numpy as np
import pathlib
import pytest
import typing

import iree.turbine.aot as aot
import torch

@dataclasses.dataclass(frozen=True, kw_only=True)
class Formula:
    """
    Represents the formula:

    (coeff * numpy.random.rand(*shape) + offset).astype(dtype)
    """
    shape: typing.Tuple[int, ...]
    dtype: type = np.dtype("float32")
    coeff: numbers.Number = 1
    offset: numbers.Number = 0

    def numpy(self):
        return (self.coeff * np.random.rand(*self.shape) + self.offset).astype(self.dtype)

    def torch(self):
        return torch.from_numpy(self.numpy())

    def toJSONEncoder(self):
        """
        Ensure all fields in dataclass are able to be encoded into JSON.

        self.dtype:type cannot be encoded into JSON
        """
        return {"Formula": {"shape": self.shape, "dtype": self.dtype.name, "coeff": self.coeff, "offset": self.offset}}

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Formula):
            return obj.toJSONEncoder()
        return super().default(obj)

@dataclasses.dataclass(frozen=True)
class ExportVariant:
    """ExportVariant class.

    This is just a wrapper around callable. Useful to denote
    a specific type of function inside a class.
    """
    function: callable
    name: str

    def __call__(self, *args, **kwargs):
        return self.function(self, *args, **kwargs)

def export_variant(function=None, seed=0):
    """
    Args:
        function: This is a function that provides the arguments to
            turbine's FxProgramsBuilder.export_program function.
            If multiple export_variants are defined per class, it is necessary
            to return name to disambiguate between different functions.
            See up-to date documentation for parameters expected by
            FxProgramsBuilder.export_program
            
            https://github.com/iree-org/iree-turbine/blob/51a22c97945b049fc79816f38108b5a8f12d2610/iree/turbine/aot/fx_programs.py#L174-L179

            By default name will be the name of the function which is annotated with export_variant
        seed: parameter used in numpy.random.seed
    """

    if function:

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            np.random.seed(seed)
            export_kwargs = function(*args, **kwargs)
            np.random.seed(0)
            if "name" not in export_kwargs:
                export_kwargs["name"] = function.__name__
            return export_kwargs

        return ExportVariant(wrapper, wrapper.__name__)

    return functools.partial(export_variant, seed=seed)


class TestProgramsBuilder(aot.FxProgramsBuilder):
    def __init__(self, *args, root_generated=None, **kwargs):
        if not root_generated:
            self.root_generated = pathlib.Path("generated")
            self.root_generated.mkdir(exist_ok=True)
        self.mlir_folder = None

        super().__init__(*args, **kwargs)

        name = self.root_module._get_name()
        self.mlir_folder = self.root_generated / name
        self.mlir_folder.mkdir(exist_ok=True)


    def generate_mlir_module(self):
        for attr_name in dir(self.root_module):
            attr = getattr(self.root_module, attr_name)
            if isinstance(attr, ExportVariant):
                export_variant = attr
                export_kwargs = export_variant()

                @self.export_program(**export_kwargs)
                def entry_point(module, *args, **kwargs):
                    return module.forward(*args, **kwargs)

        exported_module = aot.export(self)
        name = self.root_module._get_name()
        mlir_file = self.mlir_folder / f"{name}.mlir"
        exported_module.save_mlir(mlir_file)

    def generate_correctness_test(self, func, marker):
        args, kwargs = func()

        input = {}
        input["args"] = args
        input["kwargs"] = kwargs
        input["marker"] = "correctness"
        input.update(marker.kwargs)

        name = func.__name__
        root_test = self.mlir_folder / name / input["marker"]
        root_test.mkdir(exist_ok=True, parents=True)

        args_torch = []
        for arg in args:
            if isinstance(arg, Formula):
                args_torch.append(arg.torch())
            else:
                args_torch.append(arg)

        kwargs_torch = {}
        for k, v in kwargs:
            kwargs_torch[k] = v.to_torch()

        expected_output = self.root_module.forward(*args_torch, **kwargs_torch)
        if type(expected_output) != list:
            expected_output = [expected_output]

        expected_results = []
        for idx, output in enumerate(expected_output):
            fname = f"expected_result_{idx}.npy"
            np.save(root_test / fname, output)
            expected_results.append(fname)

        input["expected_results"] = expected_results

        # Save each json file into its own directory
        with open(root_test / "run_module_io_flags.json", "w") as config:
            json.dump(input, config, indent=4, cls=CustomJSONEncoder)
            print("", file=config)

    def generate_benchmark_test(self, func, marker):
        args, kwargs = func()

        input = {}
        input["args"] = args
        input["kwargs"] = kwargs
        input["marker"] = "benchmark"
        input.update(marker.kwargs)

        name = func.__name__
        root_test = self.mlir_folder / name / input["marker"]
        root_test.mkdir(exist_ok=True, parents=True)

        # Save each json file into its own directory
        with open(root_test / "run_module_io_flags.json", "w") as config:
            json.dump(input, config, indent=4, cls=CustomJSONEncoder)
            print("", file=config)

    def generate_config_files(self):
        for attr_name in dir(self.root_module):
            attr = getattr(self.root_module, attr_name)
            if not callable(attr):
                continue

            func = attr
            if not hasattr(func, "pytestmark"):
                continue

            markers = func.pytestmark
            for marker in markers:
                if marker.name == "correctness_test":
                    self.generate_correctness_test(func, marker)
                if marker.name == "benchmark_test":
                    self.generate_benchmark_test(func, marker)

    def generate_tests(self):
        self.generate_mlir_module()
        self.generate_config_files()


class AB(torch.nn.Module):
    def forward(self, left, right):
        return left @ right

    @export_variant(seed=0)
    def float32(self):
        left = Formula(shape=(64,64)).torch()
        right = Formula(shape=(64,64)).torch()
        export_kwargs = {"args" : (left, right)}
        return export_kwargs

    @export_variant(seed=0)
    def float16(self):
        left = Formula(shape=(64,64), dtype=np.dtype("float16")).torch()
        right = Formula(shape=(64,64), dtype=np.dtype("float16")).torch()
        export_kwargs = {"args" : (left, right)}
        return export_kwargs

    @pytest.mark.benchmark_test(entry_point="float32", seed=0)
    @pytest.mark.correctness_test(entry_point="float32", seed=0)
    def test_float32(self):
        args = [Formula(shape=(64, 64), dtype=np.dtype("float32"))] * 2
        return args, {}

    @pytest.mark.correctness_test(entry_point="float16", seed=0)
    def test_float16(self):
        args = [Formula(shape=(64, 64), dtype=np.dtype("float16"))] * 2
        return args, {}


TestProgramsBuilder(AB()).generate_tests()
