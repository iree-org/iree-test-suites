from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import torch
import iree.turbine.aot as aot


def camel_to_snake(name):
    result = ""
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result += "_"
        result += char.lower()
    return result


class TestGenerator(ABC, torch.nn.Module):
    def __init__(self, *args, name=None, export_kwargs=None, **kwargs):
        assert name
        self.export_kwargs = export_kwargs
        if not self.export_kwargs:
            self.export_kwargs = {}
        self.args = args
        self.kwargs = kwargs
        self.path = Path(camel_to_snake("Test" + name))
        self.path.mkdir(exist_ok=True)
        self.test_config = {}
        super().__init__()

    def save_mlir(self, *args):
        # TODO: export_kwargs should influence name?
        exported_module = aot.export(self, *args, **self.export_kwargs)
        exported_module.save_mlir(self.path / "test.mlir")

    def save_inputs(self, *args):
        inputs = []
        for idx, input in enumerate(args):
            fname = f"input{idx}.npy"
            path = self.path / fname
            np.save(path, input)
            inputs.append(fname)
        self.test_config["inputs"] = inputs

    def save_results(self, *args):
        expected_outputs = []
        for idx, result in enumerate(args):
            fname = f"result{idx}.npy"
            path = self.path / fname
            np.save(path, result)
            expected_outputs.append(fname)
        self.test_config["expected_outputs"] = expected_outputs

    def save_config(self):
        with open(self.path / "run_module_io_flags.txt", "w") as config:
            for file in self.test_config["inputs"]:
                print("--input=@" + str(file), file=config)
            for file in self.test_config["expected_outputs"]:
                print("--expected_output=@" + str(file), file=config)

    def generate_test(self, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
        """
        The default values for rtol, atol, and equal_nan are taken from the numpy's
        allclose default values here.

        https://github.com/numpy/numpy/blob/2f7fe64b8b6d7591dd208942f1cc74473d5db4cb/numpy/_core/numeric.py#L2254
        """
        inputs = self.generate_inputs()
        self.save_mlir(*inputs)
        expected_results = self.generate_expected_value(*inputs)
        self.save_inputs(*inputs)
        self.save_results(*expected_results)
        self.save_config()

    def generate_expected_value(self, *args):
        results = self.forward(*args)
        return [results]

    def generate_inputs(self):
        return self.args

    @abstractmethod
    def forward(self, *args):
        ...


class AB(TestGenerator):
    def forward(self, left, right):
        return left @ right


class ATB(TestGenerator):
    def forward(self, left, right):
        return left.t() @ right


class ABT(TestGenerator):
    def forward(self, left, right):
        return left @ right.t()


class ABplusC(TestGenerator):
    def forward(self, A, B, C):
        return A @ B + C


class ReluABPlusC(TestGenerator):
    def forward(self, A, B, C):
        return torch.relu(A @ B + C)


class GeluABPlusC(TestGenerator):
    def forward(self, A, B, C):
        return torch.ops.aten.gelu.default(A @ B + C)


# TODO: itertools
for dtype in [torch.float32, torch.float16]:
    for cls in [AB]:
        inputs = (torch.rand(64, 64, dtype=dtype), torch.rand(64, 64, dtype=dtype))
        dyn_dim = torch.export.Dim("N")
        dynamic_shapes = {"left": {0: dyn_dim}, "right": {1: dyn_dim}}
        instance = cls(
            *inputs,
            name=cls.__name__ + str(dtype),
            export_kwargs={"dynamic_shapes": dynamic_shapes},
        )
        instance.generate_test()
    for cls in [ATB, ABT]:
        inputs = (torch.rand(64, 64, dtype=dtype), torch.rand(64, 64, dtype=dtype))
        instance = cls(*inputs, name=cls.__name__ + str(dtype))
        instance.generate_test()
    for cls in [ABplusC, ReluABPlusC]:
        inputs = (
            torch.rand(64, 64, dtype=dtype) * 2 - 1,
            torch.rand(64, 64, dtype=dtype) * 2 - 1,
            torch.rand(64, 64, dtype=dtype) * 2 - 1,
        )
        instance = cls(*inputs, name=cls.__name__ + "_" + str(dtype))
        instance.generate_test()
    for cls in [GeluABPlusC]:
        inputs = (
            torch.rand(64, 64, dtype=dtype),
            torch.rand(64, 64, dtype=dtype),
            torch.rand(64, 64, dtype=dtype),
        )
        instance = cls(*inputs, name=cls.__name__ + "_" + str(dtype))
        instance.generate_test(atol=1.0e-4)
