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

from utils import CustomJSONEncoder, Formula


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
    def __init__(self, root_module, directory=None):
        """
        Args:
            root_module (torch.nn.Module): module to be exported.
            kwargs: keyword arguments forwarded to the base class.
            directory (None | str | pathlib.Path): path to the
                generated directory that will hold generated tests.
                Defaults to "generated".

        Side-effects:
            Creates directory ${directory}.
            Creates directory ${directory}/${root_module} which holds the
                mlir module generated.
        """
        if not directory:
            directory = "generated"

        if isinstance(directory, str):
            directory = pathlib.Path(directory)

        self.directory = directory
        self.directory.mkdir(exist_ok=True)
        self.mlir_folder = None

        super().__init__(root_module)

        name = self.root_module._get_name()
        self.mlir_folder = self.directory / name
        self.mlir_folder.mkdir(exist_ok=True)

    def generate_mlir_module(self, module, directory, name=None):
        """
        Export module to ${directory}/${name}.mlir.

        Args:
            module (torch.nn.Module): module to be exported.
            directory (pathlib.Path): path to the directory where the mlir file will be saved.
            name (None | str): name of the mlir file. Defaults to module._get_name().

        Returns:
            mlir_file (pathlib.Path): path to the created mlir file.

        Side-effects:
            Creates file ${directory}/${name}.mlir.
        """
        if not name:
            name = module._get_name()

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, ExportVariant):
                export_variant = attr
                export_kwargs = export_variant()

                import functools

                @self.export_program(**export_kwargs)
                def entry_point(module, *args, **kwargs):
                    return module.forward(*args, **kwargs)

        exported_module = aot.export(self)
        mlir_file = directory / f"{name}.mlir"
        exported_module.save_mlir(mlir_file)
        return mlir_file

    def generate_config_files(self, module, directory):
        """
        Args:
            module (torch.nn.Module): the module containing the tests.
            directory (pathlib.Path): path to the directory where the mlir file will be saved.

        Side-effects:
            Creates directories ${directory}/${tests}/{correctness,benchmark}
            which correspond to tests.
        """
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not callable(attr):
                continue

            func = attr
            if not hasattr(func, "pytestmark"):
                continue

            markers = func.pytestmark
            for marker in markers:
                if marker.name == "correctness_test":
                    self.generate_correctness_test(directory, module, func, marker)
                if marker.name == "benchmark_test":
                    self.generate_benchmark_test(directory, func, marker)

    @staticmethod
    def _forward(module, *args, **kwargs):
        """
        Wrapper around module.forward that converts args and kwargs of
        type Formula to torch tensors.
        """
        args_torch = []
        for arg in args:
            if isinstance(arg, Formula):
                args_torch.append(arg.torch())
            else:
                args_torch.append(arg)

        kwargs_torch = {}
        for k, v in kwargs:
            kwargs_torch[k] = v.to_torch()

        expected_output = module.forward(*args_torch, **kwargs_torch)

        if type(expected_output) != list:
            expected_output = [expected_output]

        return expected_output

    def generate_correctness_test(self, directory, module, func, marker):
        """
        Generate correctness tests for module in ${directory}/${func.__name__}/correctness.

        Args:
            directory (pathlib.Path): path to the directory that will contain
                the tests.
            module (torch.nn.Module): the module containing tests to be generated.
            func (callable): a function that will return the input args, kwargs
                needed to run the module's forward function. These are expected to
                be Formula, not torch tensors.
            marker (pytest.Mark): A pytest marker containing meta-information needed
                for generating and running the test. The correctness test accepts
                the following keyword arguments in the correctness_test mark:
                    * seed (int): Seed sent to numpy's rng.
                    * entry_point (str): Function being tested.
                    * rtol (None | float): Relative tolerance.
                    * atol (None | float): Absolute tolerance.
                    * equal_nan (None | bool): whether nan's are equal to each other.
                For rtol, atol, and equal_nan, the default values match those of np.allclose.

        Side-effects:
            Creates ${directory}/${func.__name__}/correctness folder with files:
                * expected_result_${idx}.npy
                * run_module_io.json
        """
        args, kwargs = func()

        input = {}
        input["args"] = args
        input["kwargs"] = kwargs
        input["marker"] = marker

        name = func.__name__
        root_test = directory / name / input["marker"].name
        np.random.seed(marker.kwargs.get("seed", 0))
        root_test.mkdir(exist_ok=True, parents=True)

        expected_output = TestProgramsBuilder._forward(module, *args, **kwargs)

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

    def generate_benchmark_test(self, directory, func, marker):
        """
        Generate benchmark tests for the current module in ${directory}/${func.__name__}/benchmark

        Args:
            directory (pathlib.Path): path to the directory that will contain
                the tests.
            func (callable): a function that will return the input args, kwargs
                needed to run the module's forward function. These are expected
                to be Formula not torch tensors.
            marker (pytest.Mark): A pytest marker containing meta-information needed
                for generating and running the test. The benchmark test accepts
                the following keyword arguments in the benchmark_test mark:
                    * seed (int): Seed sent to numpy's rng.
                    * entry_point (str): Function being benchmarked.

        Side-effects:
            Creates ${directory}/${func.__name__}/benchmark folder with file:
                * run_module_io.json
        """
        args, kwargs = func()

        input = {}
        input["args"] = args
        input["kwargs"] = kwargs
        input["marker"] = marker

        name = func.__name__
        root_test = directory / name / input["marker"].name
        root_test.mkdir(exist_ok=True, parents=True)

        # Save each json file into its own directory
        with open(root_test / "run_module_io_flags.json", "w") as config:
            json.dump(input, config, indent=4, cls=CustomJSONEncoder)
            print("", file=config)


    def generate_tests(self):
        """
        Generates tests in the current root_module.

        Side-effects:
            Creates directory ${directory}.
        """
        self.generate_mlir_module(self.root_module, self.mlir_folder)
        self.generate_config_files(self.root_module, self.mlir_folder)


class AB(torch.nn.Module):
    def forward(self, left, right):
        return left @ right

    @export_variant(seed=0)
    def float32(self):
        left = Formula(shape=(64, 64)).torch()
        right = Formula(shape=(64, 64)).torch()
        export_kwargs = {"args": (left, right)}
        return export_kwargs

    @export_variant(seed=0)
    def float16(self):
        left = Formula(shape=(64, 64), dtype=np.dtype("float16")).torch()
        right = Formula(shape=(64, 64), dtype=np.dtype("float16")).torch()
        export_kwargs = {"args": (left, right)}
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

class AB_bfloat16(torch.nn.Module):
    def forward(self, left, right):
        left = left.to(torch.bfloat16)
        right = right.to(torch.bfloat16)
        res = left @ right
        return res.to(torch.float32)

    @export_variant(seed=0)
    def from_float32(self):
        arg = Formula(shape=(64, 64), dtype=np.dtype("float32"))
        args = (arg.torch(), arg.torch())

        dynamic_shapes = {"args":
            ((64, torch.export.Dim("K")),
            (torch.export.Dim("K"), 64))
        }

        return {"args": args, "dynamic_shapes": dynamic_shapes}

    @pytest.mark.correctness_test(entry_point="from_float32", seed=0, rtol=1e-2, atol=1e-2)
    def test_from_float32(self):
        arg = Formula(shape=(64, 64), dtype=np.dtype("float32"))
        args = (arg, arg)
        return args, {}

class ATB(torch.nn.Module):
    def forward(self, left, right):
        return left.t() @ right

    @export_variant(seed=3)
    def float32(self):
        left = Formula(shape=(64, 64), dtype=np.dtype("float32")).torch()
        right = Formula(shape=(64, 64), dtype=np.dtype("float32")).torch()
        return {"args": (left, right)}

    @pytest.mark.correctness_test(entry_point="float32", seed=3)
    def test_float_32(self):
        left = Formula(shape=(64, 64), dtype=np.dtype("float32"))
        right = Formula(shape=(64, 64), dtype=np.dtype("float32"))
        return (left, right), {}

    @export_variant(seed=5)
    def float16(self):
        left = Formula(shape=(64, 64), dtype=np.dtype("float16")).torch()
        right = Formula(shape=(64, 64), dtype=np.dtype("float16")).torch()
        return {"args": (left, right)}

    @pytest.mark.correctness_test(entry_point="float16", seed=5)
    def test_float_16(self):
        left = Formula(shape=(64, 64), dtype=np.dtype("float16"))
        right = Formula(shape=(64, 64), dtype=np.dtype("float16"))
        return (left, right), {}

class ABT(torch.nn.Module):
    def forward(self, left, right):
        return left @ right.t()

    @export_variant(seed=5)
    def float32(self):
        left = Formula(shape=(64, 64), dtype=np.dtype("float32")).torch()
        right = Formula(shape=(64, 64), dtype=np.dtype("float32")).torch()
        return {"args": (left, right)}

    @pytest.mark.correctness_test(entry_point="float32", seed=5)
    def test_float_32(self):
        left = Formula(shape=(64, 64), dtype=np.dtype("float32"))
        right = Formula(shape=(64, 64), dtype=np.dtype("float32"))
        return (left, right), {}

    @export_variant(seed=6)
    def float16(self):
        left = Formula(shape=(64, 64), dtype=np.dtype("float16")).torch()
        right = Formula(shape=(64, 64), dtype=np.dtype("float16")).torch()
        return {"args": (left, right)}

    @pytest.mark.correctness_test(entry_point="float16", seed=6)
    def test_float_16(self):
        left = Formula(shape=(64, 64), dtype=np.dtype("float16"))
        right = Formula(shape=(64, 64), dtype=np.dtype("float16"))
        return (left, right), {}

class ABPlusC(torch.nn.Module):
    def forward(self, A, B, C):
        return A @ B + C

    @export_variant(seed=7)
    def float32(self):
        A = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=2, offset=-1).torch()
        B = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=2, offset=-1).torch()
        C = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=2, offset=-1).torch()
        return {"args": (A, B, C)}

    @pytest.mark.correctness_test(entry_point="float32", seed=7)
    def test_float32(self):
        A = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=2, offset=-1)
        B = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=2, offset=-1)
        C = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=2, offset=-1)
        return (A, B, C), {}

    @export_variant(seed=8)
    def float16(self):
        A = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=2, offset=-1).torch()
        B = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=2, offset=-1).torch()
        C = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=2, offset=-1).torch()
        return {"args": (A, B, C)}

    @pytest.mark.correctness_test(entry_point="float16", seed=8)
    def test_float16(self):
        A = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=2, offset=-1)
        B = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=2, offset=-1)
        C = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=2, offset=-1)
        return (A, B, C), {}

class ReluABPlusC(torch.nn.Module):
    def forward(self, A, B, C):
        return torch.relu(A @ B + C)

    @export_variant(seed=9)
    def float32(self):
        A = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=2, offset=-1).torch()
        B = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=2, offset=-1).torch()
        C = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=2, offset=-1).torch()
        return {"args": (A, B, C)}

    @pytest.mark.correctness_test(entry_point="float32", seed=9)
    def test_float32(self):
        A = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=2, offset=-1)
        B = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=2, offset=-1)
        C = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=2, offset=-1)
        return (A, B, C), {}

    @export_variant(seed=10)
    def float16(self):
        A = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=2, offset=-1).torch()
        B = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=2, offset=-1).torch()
        C = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=2, offset=-1).torch()
        return {"args": (A, B, C)}

    @pytest.mark.correctness_test(entry_point="float16", seed=8)
    def test_float16(self):
        A = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=2, offset=-1)
        B = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=2, offset=-1)
        C = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=2, offset=-1)
        return (A, B, C), {}


class GeluABPlusC(torch.nn.Module):
    def forward(self, A, B, C):
        return torch.ops.aten.gelu.default(A @ B + C)

    @export_variant(seed=11)
    def float32(self):
        A = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=0.1, offset=-0.05).torch()
        B = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=0.1, offset=-0.05).torch()
        C = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=0.1, offset=-0.05).torch()
        return {"args": (A, B, C)}

    @export_variant(seed=12)
    def float16(self):
        A = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=0.1, offset=-0.05).torch()
        B = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=0.1, offset=-0.05).torch()
        C = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=0.1, offset=-0.05).torch()
        return {"args": (A, B, C)}

    @pytest.mark.correctness_test(entry_point="float32", seed=11, atol=1e-5, rtol=1e-4)
    def test_float32(self):
        A = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=0.1, offset=-0.05)
        B = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=0.1, offset=-0.05)
        C = Formula(shape=(64, 64), dtype=np.dtype("float32"), coeff=0.1, offset=-0.05)
        return (A, B, C), {}

    @pytest.mark.correctness_test(entry_point="float16", seed=12, atol=1e-3, rtol=1e-3)
    def test_float16(self):
        A = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=0.1, offset=-0.05)
        B = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=0.1, offset=-0.05)
        C = Formula(shape=(64, 64), dtype=np.dtype("float16"), coeff=0.1, offset=-0.05)
        return (A, B, C), {}


for cls in [AB, AB_bfloat16, ATB, ABT, ABPlusC, ReluABPlusC, GeluABPlusC]:
    TestProgramsBuilder(cls()).generate_tests()
