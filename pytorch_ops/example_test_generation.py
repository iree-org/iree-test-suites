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

class SLogDetTest(OpTest):
    def forward(self, A):
        return torch.linalg.slogdet(A)

    def generate_inputs(self):
        return [torch.rand(64,64)]

class TorchAtenNormScalarTest(OpTest):
    def forward(self, A):
        return torch.ops.aten.norm.Scalar(A, 0.5)

    def generate_inputs(self):
        return [torch.rand(64)]

# torch.aten.hann_window.periodic
class TorchAtenHannWindowPeriodicTest(OpTest):
    def forward(self, *args):
        return torch.ops.aten.hann_window.periodic(128, True)

    def generate_inputs(self):
        return []

class TorchAtenRenormTest(OpTest):
    def forward(self, x):
        return torch.ops.aten.renorm(x, 2.0, 0, 1.0)

    def generate_inputs(self):
        return [torch.rand(128, 128)]

class TorchAtenAllDimTest(OpTest):
    def forward(self, x):
        return torch.ops.aten.all.dim(x, 0)

    def generate_inputs(self):
        return [torch.randint(0, 2, (128, 128), dtype=torch.bool)]

class TorchAtenTriuIndicesTest(OpTest):
    def forward(self, *args):
        return torch.ops.aten.triu_indices(8, 8, 0)

    def generate_inputs(self):
        return []

class TorchAtenKthValueTest(OpTest):
    def forward(self, x):
        return torch.ops.aten.kthvalue(x, 1, 0)  # k=1, dim=0

    def generate_inputs(self):
        return [torch.rand(128, 128)]

class TorchAtenAvgPool2dTest(OpTest):
    def forward(self, x):
        return torch.ops.aten.avg_pool2d(x, kernel_size=[2,2], stride=[2,2], padding=[0,0])

    def generate_inputs(self):
        return [torch.rand(1, 64, 32, 32)]  # [batch_size, channels, height, width]

class TorchAtenTrilIndicesTest(OpTest):
    def forward(self, *args):
        return torch.ops.aten.tril_indices(128, 128, 0)

    def generate_inputs(self):
        return []

def main():
    tests = [
        MatMulOpTest("test_matmul_64x64"),
        TrilinearOpTest("test_trilinear_64x64"),
        UnfoldTest("test_unfold_128"),
        SLogDetTest("test_slogdet_64x64"),
        TorchAtenNormScalarTest("test_torch_aten_norm_scalar_64"),
        TorchAtenHannWindowPeriodicTest("test_torch_aten_hann_window_periodic_128"),
        TorchAtenRenormTest("test_torch_aten_renorm_128x128"),
        TorchAtenAllDimTest("test_torch_aten_all_dim_128x128"),
        TorchAtenTriuIndicesTest("test_torch_aten_triu_indices_test_128"),
        TorchAtenKthValueTest("test_torch_aten_kth_value_test"),
        TorchAtenAvgPool2dTest("test_torch_aten_avg_pool_2d"),
        TorchAtenTrilIndicesTest("test_torch_aten_tril_indices"),
    ]

    for test in tests:
        test.generate_test()


if "__main__" == __name__:
    main()
