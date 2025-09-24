import iree.runtime as ireert
import numpy as np
import iree.turbine.aot as aot
import torch
from pathlib import Path


class OpTest(torch.nn.Module):

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.path = Path(name)
        self.path.mkdir(exist_ok=True)

    def save_mlir(self, *args):
        exported_module = aot.export(self, *args)
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

    def generate_expected_value(self, *args):
        results = self.forward(*args)
        return [results]

    def generate_inputs(self): ...

    def forward(self, *args): ...


class MatMulOpTest(OpTest):

    def forward(self, *args):
        left, right = args
        result = left @ right
        return result

    def generate_inputs(self):
        input0 = torch.rand(64, 64)
        input1 = torch.rand(64, 64)
        return input0, input1


class TrilinearOpTest(OpTest):
    def forward(self, *args):
        a, b, c = args
        result = torch.ops.aten._trilinear(
            a, b, c, expand1=[], expand2=[], expand3=[], sumdim=[], unroll_dim=0
        )
        return result

    def generate_inputs(self):
        input0 = torch.rand(64, 64)
        input1 = torch.rand(64, 64)
        input2 = torch.rand(64, 64)
        return input0, input1, input2


class UnfoldTest(OpTest):
    def forward(self, *args):
        return args[0].unfold(0, 2, 1)

    def generate_inputs(self):
        return [torch.rand(128)]


def main():
    tests = [
        MatMulOpTest("test_matmul_64x64"),
        TrilinearOpTest("test_trilinear_64x64"),
        UnfoldTest("test_unfold_128"),
    ]

    for test in tests:
        test.generate_test()


if "__main__" == __name__:
    main()
