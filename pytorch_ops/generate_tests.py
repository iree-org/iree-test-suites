from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import json

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
    def __init__(self, *args, name=None, **kwargs):
        assert name
        self.args = args
        self.kwargs = kwargs
        self.path = Path(camel_to_snake("Test" + name))
        self.path.mkdir(exist_ok=True)
        self.test_config = {}
        super().__init__()

    def get_export_kwargs(self):
        # Default implementation is that exported kwargs is an empty dictionary.
        return {}

    def save_mlir(self, *args):
        export_kwargs = self.get_export_kwargs()
        exported_module = aot.export(self, *args, **export_kwargs)
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
        observed_outputs = []
        for idx, result in enumerate(args):
            fname_expected = f"expected_result{idx}.npy"
            fname_observed = f"observed_result{idx}.npy"
            path = self.path / fname_expected
            np.save(path, result)
            expected_outputs.append(fname_expected)
            observed_outputs.append(fname_observed)
        self.test_config["expected_outputs"] = expected_outputs
        self.test_config["observed_outputs"] = observed_outputs

    def save_config(self):
        with open(self.path / "run_module_io_flags.json", "w") as config:
            json.dump(self.test_config, config, indent=4)
            print("", file=config)

    def generate_test(self, rtol=1e-05, atol=1e-08, equal_nan=False):
        """
        The default values for rtol, atol, and equal_nan come from the
        numpy.allclose's default values listed in the link below.
        https://numpy.org/devdocs/reference/generated/numpy.allclose.html

        These values will be saved in the json file and be used when
        running the test.
        """
        inputs = self.generate_inputs()
        self.test_config["rtol"] = rtol
        self.test_config["atol"] = atol
        self.test_config["equal_nan"] = equal_nan
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


@dataclass
class RandomSession:
    generator: torch.Generator = torch.Generator()
    seed: int = 0

    def __post_init__(self):
        self.generator.manual_seed(self.seed)

    def rand(self, *args, **kwargs):
        if kwargs.get("generator"):
            raise ValueError("RandomSession uses its own generator")
        kwargs["generator"] = self.generator
        return torch.rand(*args, **kwargs)


class AB(TestGenerator):
    def forward(self, left, right):
        return left @ right

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

    def get_export_kwargs(self):
        dyn_dim = torch.export.Dim("N")
        dynamic_shapes = {"left": {0: dyn_dim}, "right": {1: dyn_dim}}
        return {"dynamic_shapes": dynamic_shapes}


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


# TODO: Future PR, itertools.product where it makes sense
for dtype in [torch.float32]:
    for cls in [AB_bfloat16]:
        random = RandomSession()
        inputs = (random.rand(64, 64, dtype=dtype), random.rand(64, 64, dtype=dtype))
        instance = cls(*inputs, name=cls.__name__)
        instance.generate_test(atol=1e-2, rtol=1e-2)
for dtype in [torch.float32, torch.float16]:
    for cls in [AB]:
        random = RandomSession()
        inputs = (random.rand(64, 64, dtype=dtype), random.rand(64, 64, dtype=dtype))
        instance = cls(*inputs, name=cls.__name__ + str(dtype))
        instance.generate_test()
    for cls in [ATB, ABT]:
        random = RandomSession()
        inputs = (random.rand(64, 64, dtype=dtype), random.rand(64, 64, dtype=dtype))
        instance = cls(*inputs, name=cls.__name__ + str(dtype))
        instance.generate_test()
    for cls in [ABplusC, ReluABPlusC, GeluABPlusC]:
        random = RandomSession()
        inputs = (
            random.rand(64, 64, dtype=dtype) * 2 - 1,
            random.rand(64, 64, dtype=dtype) * 2 - 1,
            random.rand(64, 64, dtype=dtype) * 2 - 1,
        )
        instance = cls(*inputs, name=cls.__name__ + "_" + str(dtype))
        instance.generate_test()
    for cls in [GeluABPlusC]:
        random = RandomSession()
        inputs = (
            random.rand(64, 64, dtype=dtype) * 0.1 - 0.05,
            random.rand(64, 64, dtype=dtype) * 0.1 - 0.05,
            random.rand(64, 64, dtype=dtype) * 0.1 - 0.05,
        )
        instance = cls(*inputs, name=cls.__name__ + "_" + str(dtype))
        if cls == GeluABPlusC:
            match dtype:
                case torch.float32:
                    instance.generate_test(atol=1e-5, rtol=1e-4)
                case torch.float16:
                    instance.generate_test(atol=1e-3, rtol=1e-3)
