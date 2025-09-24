import iree.runtime as ireert
import numpy as np
import iree.turbine.aot as aot
import torch
from pathlib import Path

# Define the `nn.Module` to export.
class LinearModule(torch.nn.Module):
  def forward(self, left, right):
    result = (left @ right) 
    return result

linear_module = LinearModule()

class MatMulOpTest:

    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.path = Path(name)
        self.path.mkdir()
    
    def save_mlir(self, *args):
        exported_module = aot.export(self.module, *args)
        exported_module.save_mlir(self.path / "test.mlir")

    def save_inputs(self, *args):
        for idx, input in enumerate(args):
            path = self.path / f"input{idx}.npy"
            np.save(path, input)

    def save_results(self, *args):
        for idx, result in enumerate(args):
            path = self.path / f"result{idx}.npy"
            np.save(path, result)

    def generate_test(self):
        inputs = self.generate_inputs()
        self.save_mlir(*inputs)
        results = self.generate_expected_value(*inputs)

        self.save_inputs(*inputs)
        self.save_results(*results)

    def generate_inputs(self):
        input0 = torch.rand(64, 64)
        input1 = torch.rand(64, 64)
        return input0, input1

    def generate_expected_value(self, *args):
        results = self.module.forward(*args)
        return [results]

test = MatMulOpTest(linear_module, "test_matmul_64x64")
test.generate_test()

