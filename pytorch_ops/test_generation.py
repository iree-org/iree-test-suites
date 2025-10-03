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

    def __init__(self, *args, **kwargs):
        self.path = Path(camel_to_snake("Test" + type(self).__name__))
        self.path.mkdir(exist_ok=True)
        self.test_config = {}
        super().__init__()

    def save_mlir(self, *args):
        exported_module = aot.export(self, *args)
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

    def generate_test(self):
        inputs = self.generate_inputs()
        self.save_mlir(*inputs)
        results = self.generate_expected_value(*inputs)

        self.save_inputs(*inputs)
        self.save_results(*results)
        self.save_config()

    def generate_expected_value(self, *args):
        results = self.forward(*args)
        return [results]

    @abstractmethod
    def generate_inputs(self): ...

    @abstractmethod
    def forward(self, *args): ...

class AB(TestGenerator):
    def forward(self, left, right):
        return left @ right

    def generate_inputs(self):
        return torch.rand(64, 64), torch.rand(64, 64)

class ATB(TestGenerator):
    def forward(self, left, right):
        return left.t() @ right

    def generate_inputs(self):
        return torch.rand(64, 64), torch.rand(64, 64)

class ABT(TestGenerator):
    def forward(self, left, right):
        return left @ right.t()

    def generate_inputs(self):
        return torch.rand(64, 64), torch.rand(64, 64)

class ABplusC(TestGenerator):
    def forward(self, A, B, C):
        return A @ B + C
    def generate_inputs(self):
        return torch.rand(64, 64), torch.rand(64, 64), torch.rand(64, 64)

class ReluABplusC(TestGenerator):
    def forward(self, A, B, C):
        return torch.relu(A @ B + C)

    def generate_inputs(self):
        return torch.rand(64, 64) * 2 - 1, torch.rand(64, 64) * 2 - 1, torch.rand(64, 64) * 2 - 1

class GeluABplusC(TestGenerator):
    def forward(self, A, B, C):
        return torch.ops.aten.gelu.default(A @ B + C)

    def generate_inputs(self):
        return torch.rand(64, 64) * 2 - 1, torch.rand(64, 64) * 2 - 1, torch.rand(64, 64) * 2 - 1


for cls in [AB, ATB, ABT, ABplusC, ReluABplusC, GeluABplusC]:
    instance = cls()
    instance.generate_test()
