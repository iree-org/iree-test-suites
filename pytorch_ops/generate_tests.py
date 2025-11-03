from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import json
import functools

import torch
import iree.turbine.aot as aot


def camel_to_snake(name):
    result = ""
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result += "_"
        result += char.lower()
    return result


class TestGenerator(torch.nn.Module, ABC):
    def __init__(self):
        self.test_config = {}
        self.generator = None
        super().__init__()

    def get_export_kwargs(self):
        # Default implementation is that exported kwargs is an empty dictionary.
        return {}

    def save_mlir(self, path, *args, **_kwargs):
        export_kwargs = self.get_export_kwargs()
        exported_module = aot.export(self, *args, **export_kwargs)
        exported_module.save_mlir(path / "test.mlir")

    def save_inputs(self, root_path, test_config, *args):
        inputs = []
        for idx, input in enumerate(args):
            fname = f"input{idx}.npy"
            path = root_path / fname
            np.save(path, input)
            inputs.append(fname)
        test_config["inputs"] = inputs
        return test_config

    def save_results(self, root_path, test_config, *args):
        expected_outputs = []
        observed_outputs = []
        for idx, result in enumerate(args):
            fname_expected = f"expected_result{idx}.npy"
            fname_observed = f"observed_result{idx}.npy"
            path = root_path / fname_expected
            np.save(path, result)
            expected_outputs.append(fname_expected)
            observed_outputs.append(fname_observed)
        test_config["expected_outputs"] = expected_outputs
        test_config["observed_outputs"] = observed_outputs
        return test_config

    def save_config(self, root_path, test_config):
        with open(root_path / "run_module_io_flags.json", "w") as config:
            json.dump(test_config, config, indent=4)
            print("", file=config)

    def generate_tests(self):
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue

            attr = getattr(self, name)
            if not callable(attr):
                continue

            test_config = {}
            pathname = f"test_{camel_to_snake(self._get_name())}_{name[5:]}"
            path = Path(pathname)
            path.mkdir(exist_ok=True)
            inputs_to_forward = attr

            args, kwargs, test_kwargs = inputs_to_forward()
            test_config["rtol"] = test_kwargs["rtol"]
            test_config["atol"] = test_kwargs["atol"]
            test_config["equal_nan"] = test_kwargs["equal_nan"]

            self.save_mlir(path, *args, **kwargs)
            expected_results = self.generate_expected_value(*args, **kwargs)
            test_config = self.save_inputs(path, test_config, *args)
            test_config = self.save_results(path, test_config, *expected_results)
            self.save_config(path, test_config)

    def generate_expected_value(self, *args):
        results = self.forward(*args)
        return [results]

    @abstractmethod
    def forward(self, *args):
        ...

    def rand(self, *args, **kwargs):
        if kwargs.get("generator"):
            raise ValueError("Set generator with parameter to test function.")
        kwargs["generator"] = self.generator
        return torch.rand(*args, **kwargs)


def test(
    function=None, generator=None, seed=0, rtol=1e-05, atol=1e-08, equal_nan=False
):
    """
    Decorator used to denote that a particular method corresponds
    to test input used in the forward method.

    The default values for rtol, atol, and equal_nan come from the
    numpy.allclose's default values listed in the link below.
    https://numpy.org/devdocs/reference/generated/numpy.allclose.html

    The generator will be seeded with seed and later be passed on
    to a torch.rand wrapper.

    These values will be saved in the json file and be used when
    running the test.
    """
    if not generator:
        generator = torch.Generator()

    test_kwargs = {"rtol": rtol, "atol": atol, "equal_nan": equal_nan}

    if function:

        def wrapper(self):
            self.generator = generator
            self.generator.manual_seed(seed)
            args, kwargs = function(self)
            self.generator = None
            return args, kwargs, test_kwargs

        return wrapper

    return functools.partial(test, **test_kwargs)


class AB(TestGenerator):
    def forward(self, left, right):
        return left @ right

    @test(seed=0)
    def test_float32(self):
        left = self.rand(64, 64, dtype=torch.float32)
        right = self.rand(64, 64, dtype=torch.float32)
        return ((left, right), {})

    @test(seed=1)
    def test_float16(self):
        left = self.rand(64, 64, dtype=torch.float16)
        right = self.rand(64, 64, dtype=torch.float16)
        return ((left, right), {})

    def get_export_kwargs(self):
        dyn_dim = torch.export.Dim("N")
        dynamic_shapes = {"left": {0: dyn_dim}, "right": {1: dyn_dim}}
        return {"dynamic_shapes": dynamic_shapes}


class AB_bfloat16(TestGenerator):
    def forward(self, left, right):
        left = left.to(torch.bfloat16)
        right = right.to(torch.bfloat16)
        res = left @ right
        return res.to(torch.float32)

    @test(atol=1e-2, rtol=1e-2, seed=2)
    def test_from_float32(self):
        left = self.rand(64, 64, dtype=torch.float32)
        right = self.rand(64, 64, dtype=torch.float32)
        return ((left, right), {})

    def get_export_kwargs(self):
        dyn_dim = torch.export.Dim("N")
        dynamic_shapes = {"left": {0: dyn_dim}, "right": {1: dyn_dim}}
        return {"dynamic_shapes": dynamic_shapes}


class ATB(TestGenerator):
    def forward(self, left, right):
        return left.t() @ right

    @test(seed=3)
    def test_float32(self):
        left = self.rand(64, 64, dtype=torch.float32)
        right = self.rand(64, 64, dtype=torch.float32)
        return ((left, right), {})

    @test(seed=4)
    def test_float16(self):
        left = self.rand(64, 64, dtype=torch.float16)
        right = self.rand(64, 64, dtype=torch.float16)
        return ((left, right), {})


class ABT(TestGenerator):
    def forward(self, left, right):
        return left @ right.t()

    @test(seed=5)
    def test_float32(self):
        left = self.rand(64, 64, dtype=torch.float32)
        right = self.rand(64, 64, dtype=torch.float32)
        return ((left, right), {})

    @test(seed=6)
    def test_float16(self):
        left = self.rand(64, 64, dtype=torch.float16)
        right = self.rand(64, 64, dtype=torch.float16)
        return ((left, right), {})


class ABPlusC(TestGenerator):
    def forward(self, A, B, C):
        return A @ B + C

    @test(seed=7)
    def test_float32(self):
        return (
            (
                self.rand(64, 64, dtype=torch.float32) * 2 - 1,
                self.rand(64, 64, dtype=torch.float32) * 2 - 1,
                self.rand(64, 64, dtype=torch.float32) * 2 - 1,
            ),
            {},
        )

    @test(seed=8)
    def test_float16(self):
        return (
            (
                self.rand(64, 64, dtype=torch.float16) * 2 - 1,
                self.rand(64, 64, dtype=torch.float16) * 2 - 1,
                self.rand(64, 64, dtype=torch.float16) * 2 - 1,
            ),
            {},
        )


class ReluABPlusC(TestGenerator):
    def forward(self, A, B, C):
        return torch.relu(A @ B + C)

    @test(seed=9)
    def test_float32(self):
        return (
            (
                self.rand(64, 64, dtype=torch.float32) * 2 - 1,
                self.rand(64, 64, dtype=torch.float32) * 2 - 1,
                self.rand(64, 64, dtype=torch.float32) * 2 - 1,
            ),
            {},
        )

    @test(seed=10)
    def test_float16(self):
        return (
            (
                self.rand(64, 64, dtype=torch.float16) * 2 - 1,
                self.rand(64, 64, dtype=torch.float16) * 2 - 1,
                self.rand(64, 64, dtype=torch.float16) * 2 - 1,
            ),
            {},
        )


class GeluABPlusC(TestGenerator):
    def forward(self, A, B, C):
        return torch.ops.aten.gelu.default(A @ B + C)

    @test(seed=11, atol=1e-5, rtol=1e-4)
    def test_float32(self):
        return (
            (
                self.rand(64, 64, dtype=torch.float32) * 0.1 - 0.05,
                self.rand(64, 64, dtype=torch.float32) * 0.1 - 0.05,
                self.rand(64, 64, dtype=torch.float32) * 0.1 - 0.05,
            ),
            {},
        )

    @test(seed=12, atol=1e-3, rtol=1e-3)
    def test_float16(self):
        return (
            (
                self.rand(64, 64, dtype=torch.float32) * 0.1 - 0.05,
                self.rand(64, 64, dtype=torch.float32) * 0.1 - 0.05,
                self.rand(64, 64, dtype=torch.float32) * 0.1 - 0.05,
            ),
            {},
        )


for cls in [AB, AB_bfloat16, ATB, ABT, ABPlusC, ReluABPlusC, GeluABPlusC]:
    instance = cls()
    instance.generate_tests()
