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

class TorchAtenMaxUnpool3dTest(OpTest):
    def forward(self, x, indices):
        output_size = [4, 4, 4]  # smaller output size [depth, height, width]
        kernel_size = [2, 2, 2]
        padding = [0, 0, 0]
        return torch.ops.aten.max_unpool3d(x, indices, output_size, kernel_size, padding)

    def generate_inputs(self):
        # Possibly we should not use random.
        # Right now the test fails but I am unsure if it is because of the random data.
        x = torch.rand(2, 4, 2, 2, 2)  # smaller input size [batch_size, channels, depth, height, width]
        indices = torch.randint(0, 2, (2, 4, 2, 2, 2), dtype=torch.int64)
        return [x, indices]

class TorchAtenLerpScalarTest(OpTest):
    def forward(self, start, end):
        weight = 0.5  # interpolation weight between 0 and 1
        return torch.ops.aten.lerp.Scalar(start, end, weight)

    def generate_inputs(self):
        start = torch.rand(64)  # start tensor
        end = torch.rand(64)    # end tensor
        return [start, end]

class TorchAtenLogitTest(OpTest):
    def forward(self, x):
        eps = 1e-6  # small value to avoid numerical instability
        return torch.ops.aten.logit(x, eps)

    def generate_inputs(self):
        # Generate random values between 0 and 1 since logit expects input in this range
        return [torch.rand(64)]

class TorchAtenReflectionPad2dTest(OpTest):
    def forward(self, x):
        padding = [1, 1, 1, 1]  # left, right, top, bottom padding
        return torch.ops.aten.reflection_pad2d(x, padding)

    def generate_inputs(self):
        # Input shape: [batch_size, channels, height, width]
        return [torch.rand(1, 3, 32, 32)]

class TorchAtenWeightNormInterfaceTest(OpTest):
    # test.mlir:4:26: error: 'tensor.cast' op operand type 'tensor<?x1xf32>' and result type 'tensor<16x8xf32>' are cast incompatible
    # %result0, %result1 = torch.aten._weight_norm_interface %arg0, %arg1, %int0 : !torch.vtensor<[16,8],f32>, !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[16,8],f32>, !torch.vtensor<[16,1],f32>
    #                         ^
    # test.mlir:4:26: note: see current operation: %32 = "tensor.cast"(%31) : (tensor<?x1xf32>) -> tensor<16x8xf32>
    def forward(self, v, g):
        dim = 0
        res = torch.ops.aten._weight_norm_interface(v, g, dim)
        return res[0][0], res[0][1]

    def generate_inputs(self):
        v = torch.rand(16, 8)  # weight tensor
        g = torch.rand(8)      # gain parameter
        return [v, g]

class TorchAtenPixelShuffleTest(OpTest):
    def forward(self, x):
        upscale_factor = 2  # reorganize for 2x upscaling
        return torch.ops.aten.pixel_shuffle(x, upscale_factor)

    def generate_inputs(self):
        # Input shape: [batch_size, channels*r², height, width]
        # Here r=2, so channels are multiplied by 4
        return [torch.rand(1, 12, 16, 16)]  # Will become [1, 3, 32, 32]

class TorchAtenNormalFunctionalTest(OpTest):
    # Probably hard to test since this one is supposed to
    # sample from a normal distribution.
    # We should just guarantee instead that it approximates
    # the normal distribution with some level of confidence.
    def forward(self, x):
        return torch.ops.aten.normal_functional(x, 0.0, 1.0)

    def generate_inputs(self):
        # Using a 2D tensor to test broadcasting
        x = torch.rand(32, 64)  # 2D mean tensor
        return [x]

class TorchAtenAdaptiveMaxPool2dTest(OpTest):
    # Failed: expected 1 list elements but 2 provided
    def forward(self, x):
        output_size = [8, 8]  # target output spatial size
        return torch.ops.aten.adaptive_max_pool2d(x, output_size)

    def generate_inputs(self):
        # Input shape: [batch_size, channels, height, width]
        return [torch.rand(1, 16, 32, 32)]

class TorchAtenReplicationPad2dTest(OpTest):
    def forward(self, x):
        padding = [1, 1, 1, 1]  # left, right, top, bottom padding
        return torch.ops.aten.replication_pad2d(x, padding)

    def generate_inputs(self):
        # Input shape: [batch_size, channels, height, width]
        return [torch.rand(1, 3, 32, 32)]

class TorchAtenGluTest(OpTest):
    def forward(self, x):
        dim = -1  # dimension along which to apply GLU
        return torch.ops.aten.glu(x, dim)

    def generate_inputs(self):
        # Input shape needs even number of features in the GLU dimension
        # Shape: [batch_size, features], where features must be even
        return [torch.rand(32, 64)]  # 64 features will be halved to 32 by GLU

class TorchAtenAvgPool3dTest(OpTest):
    def forward(self, x):
        kernel_size = [2, 2, 2]
        stride = [2, 2, 2]
        padding = [0, 0, 0]
        return torch.ops.aten.avg_pool3d(x, kernel_size, stride, padding)

    def generate_inputs(self):
        # Input shape: [batch_size, channels, depth, height, width]
        return [torch.rand(1, 16, 8, 8, 8)]

class TorchAtenDiagEmbedTest(OpTest):
    def forward(self, x):
        offset = 0  # diagonal offset
        dim1 = 0    # first dimension of resulting 2D matrix
        dim2 = 1    # second dimension of resulting 2D matrix
        return torch.ops.aten.diag_embed(x, offset, dim1, dim2)

    def generate_inputs(self):
        # Generate a 1D tensor that will be embedded as a diagonal
        return [torch.rand(32)]

class TorchAtenAcosTest(OpTest):
    def forward(self, x):
        return torch.ops.aten.acos(x)

    def generate_inputs(self):
        # Generate values between -1 and 1 since acos is only defined in this range
        return [torch.rand(64).mul(2).sub(1)]  # scales [0,1] to [-1,1]

class TorchAtenArgminTest(OpTest):
    # [FAILED] result[0]: metadata is 32xi64; expected that the view matches 32xsi64; expected that the view is equal to contents of a view of 32xsi64
    def forward(self, x):
        dim = 0  # dimension to reduce
        return torch.ops.aten.argmin(x, dim)

    def generate_inputs(self):
        return [torch.rand(64, 32)]  # 2D tensor to find minimum indices along dim 0

class TorchAtenMinDimTest(OpTest):
    # [FAILED] expected 1 list elements but 2 provided
    def forward(self, x):
        dim = 0  # dimension to reduce
        return torch.ops.aten.min.dim(x, dim)

    def generate_inputs(self):
        return [torch.rand(64, 32)]  # 2D tensor to find minimum values along dim 0

class TorchAtenExponentialTest(OpTest):
    # [FAILED] result[0]: element at index 0 (2.44039) does not match the expected (0.0118456); expected that the view is equal to contents of a view of 64xf32
    def forward(self, x):
        # lambda parameter defaults to 1.0
        return torch.ops.aten.exponential(x, 1.0)

    def generate_inputs(self):
        # Generate positive values since exponential distribution is only defined for x > 0
        return [torch.rand(64)]  # generates values in range [0,1]

class TorchAtenReflectionPad1dTest(OpTest):
    def forward(self, x):
        padding = [1, 1]  # left, right padding
        return torch.ops.aten.reflection_pad1d(x, padding)

    def generate_inputs(self):
        # Input shape: [batch_size, channels, width]
        return [torch.rand(1, 3, 32)]

class TorchAtenProdDimIntTest(OpTest):
    def forward(self, x):
        dim = 0  # dimension to reduce
        return torch.ops.aten.prod.dim_int(x, dim)

    def generate_inputs(self):
        return [torch.rand(64, 32)]  # 2D tensor to compute product along dim 0

class TorchAtenConvTbcTest(OpTest):
    def forward(self, input, weight, bias):
        # test.mlir:5:5: error: type of return operand 0 ('!torch.vtensor<[16,10,4],f32>') doesn't match function result type ('!torch.vtensor<[10,4,16],f32>') in function @main
        pad = 1
        return torch.ops.aten.conv_tbc(input, weight, bias, pad)

    def generate_inputs(self):
        # Input shape: [time, batch, in_channels]
        input = torch.rand(10, 4, 8)
        # Weight shape: [kernel_size, in_channels, out_channels]
        weight = torch.rand(3, 8, 16)
        # Bias shape: [out_channels]
        bias = torch.rand(16)
        return [input, weight, bias]

class TorchAtenConv1dTest(OpTest):
    def forward(self, input, weight, bias):
        stride = [1]
        padding = [0]
        dilation = [1]
        return torch.ops.aten.conv1d(input, weight, bias, stride, padding, dilation)

    def generate_inputs(self):
        # Input shape: [batch, in_channels, length]
        input = torch.rand(16, 32, 100)
        # Weight shape: [out_channels, in_channels, kernel_size]
        weight = torch.rand(64, 32, 3)
        # Bias shape: [out_channels]
        bias = torch.rand(64)
        return [input, weight, bias]

class TorchAtenConv3dTest(OpTest):
    def forward(self, input, weight, bias):
        stride = [1, 1, 1]
        padding = [0, 0, 0]
        dilation = [1, 1, 1]
        return torch.ops.aten.conv3d(input, weight, bias, stride, padding, dilation)

    def generate_inputs(self):
        # Input shape: [batch, in_channels, depth, height, width]
        input = torch.rand(2, 4, 4, 4, 4)
        # Weight shape: [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
        weight = torch.rand(8, 4, 2, 2, 2)
        # Bias shape: [out_channels]
        bias = torch.rand(8)
        return [input, weight, bias]

class TorchAtenIsInfTest(OpTest):
    def forward(self, x):
        return torch.ops.aten.isinf(x)

    def generate_inputs(self):
        # Create a tensor with some infinite values
        x = torch.tensor([float('inf'), -float('inf'), 1.0, 2.0], dtype=torch.float32)
        return [x]

class TorchAtenAdaptiveAvgPool3dTest(OpTest):
    def forward(self, x):
        output_size = [2, 2, 2]  # target output spatial size
        return torch.ops.aten._adaptive_avg_pool3d(x, output_size)

    def generate_inputs(self):
        # Input shape: [batch_size, channels, depth, height, width]
        return [torch.rand(1, 4, 4, 4, 4)]

class TorchAtenTraceTest(OpTest):
    def forward(self, x):
        return torch.ops.aten.trace(x)

    def generate_inputs(self):
        # Generate a square matrix to compute its trace
        return [torch.rand(32, 32)]

class TorchAtenAdaptiveMaxPool3dTest(OpTest):
    # test.mlir:7:26: warning: reads uninitialized values from an operand produced by a tensor.empty op. To disable this warning, pass --iree-global-opt-enable-warn-on-uninitialized-values=false
    def forward(self, x):
        output_size = [2, 2, 2]  # target output spatial size
        return torch.ops.aten.adaptive_max_pool3d(x, output_size)

    def generate_inputs(self):
        # Input shape: [batch_size, channels, depth, height, width]
        return [torch.rand(1, 4, 4, 4, 4)]

class TorchAtenLinalgCrossTest(OpTest):
    def forward(self, x, y):
        return torch.ops.aten.linalg_cross(x, y)  # cross product along last dimension

    def generate_inputs(self):
        # Generate two sets of 3D vectors
        # Shape: [N, 3] where 3 is required for cross product
        x = torch.rand(32, 3)  # 32 vectors of dimension 3
        y = torch.rand(32, 3)  # 32 vectors of dimension 3
        return [x, y]

class TorchAtenDiagonalTest(OpTest):
    def forward(self, x):
        offset = 0  # offset from the main diagonal
        dim1 = 0    # first dimension to take diagonal from
        dim2 = 1    # second dimension to take diagonal from
        return torch.ops.aten.diagonal(x, offset, dim1, dim2)

    def generate_inputs(self):
        # Generate a 3D tensor to extract diagonal from
        return [torch.rand(32, 32, 16)]

class TorchAtenCoshTest(OpTest):
    def forward(self, x):
        return torch.ops.aten.cosh(x)

    def generate_inputs(self):
        # Generate random values to compute hyperbolic cosine
        return [torch.rand(64)]

class TorchAtenRemainderTest(OpTest):
    def forward(self, x, y):
        return torch.ops.aten.remainder(x, y)

    def generate_inputs(self):
        # Generate random tensors for dividend and divisor
        x = torch.rand(64)  # dividend
        y = torch.rand(64) + 0.1  # divisor (adding 0.1 to avoid values too close to zero)
        return [x, y]

def main():
    tests = [
        #MatMulOpTest("test_matmul_64x64"),
        #TrilinearOpTest("test_trilinear_64x64"),
        #UnfoldTest("test_unfold_128"),
        #SLogDetTest("test_slogdet_64x64"),
        #TorchAtenNormScalarTest("test_torch_aten_norm_scalar_64"),
        #TorchAtenHannWindowPeriodicTest("test_torch_aten_hann_window_periodic_128"),
        #TorchAtenRenormTest("test_torch_aten_renorm_128x128"),
        #TorchAtenAllDimTest("test_torch_aten_all_dim_128x128"),
        #TorchAtenTriuIndicesTest("test_torch_aten_triu_indices_test_128"),
        #TorchAtenKthValueTest("test_torch_aten_kth_value_test"),
        #TorchAtenAvgPool2dTest("test_torch_aten_avg_pool_2d"),
        #TorchAtenTrilIndicesTest("test_torch_aten_tril_indices"),
        #TorchAtenMaxUnpool3dTest("test_torch_aten_max_unpool_3d"),
        #TorchAtenLerpScalarTest("test_torch_aten_lerp_scalar"),
        #TorchAtenLogitTest("test_torch_aten_logit"),
        #TorchAtenReflectionPad2dTest("test_torch_aten_reflection_pad_2d"),
        #TorchAtenWeightNormInterfaceTest("test_torch_aten_weight_norm_interface"),
        #TorchAtenPixelShuffleTest("test_torch_aten_pixel_shuffle")
        #TorchAtenNormalFunctionalTest("test_torch_aten_normal_functional"),
        #TorchAtenAdaptiveMaxPool2dTest("test_torch_aten_adaptive_max_pool_2d"),
        #TorchAtenReplicationPad2dTest("test_torch_aten_replication_pad_2d"),
        #TorchAtenGluTest("test_torch_aten_glu"),
        #TorchAtenAvgPool3dTest("test_torch_aten_avg_pool_3d"),
        #TorchAtenDiagEmbedTest("test_torch_aten_diag_embed"),
        #TorchAtenAcosTest("test_torch_aten_acos"),
        #TorchAtenArgminTest("test_torch_aten_argmin"),
        #TorchAtenMinDimTest("test_torch_aten_min_dim"),
        #TorchAtenExponentialTest("test_torch_aten_exponential"),
        #TorchAtenReflectionPad1dTest("test_torch_aten_reflection_pad1d"),
        #TorchAtenProdDimIntTest("test_torch_aten_prod_int"),
        #TorchAtenConvTbcTest("test_torch_aten_convtbc"),
        #TorchAtenConv1dTest("test_torch_aten_conv1d"),
        #TorchAtenConv3dTest("test_torch_aten_conv3d"),
        #TorchAtenIsInfTest("test_torch_aten_isinf"),
        #TorchAtenAdaptiveAvgPool3dTest("test_torch_aten_adaptive_avg_pool3d"),
        #TorchAtenTraceTest("test_torch_aten_trace"),
        #TorchAtenAdaptiveMaxPool3dTest("test_torch_aten_adaptive_max_pool3d"),
        #TorchAtenLinalgCrossTest("test_torch_aten_linalg_cross"),
        #TorchAtenDiagonalTest("test_torch_aten_diagonal"),
        #TorchAtenCoshTest("test_torch_aten_cosh"),
        TorchAtenRemainderTest("test_torch_aten_remainder"),
    ]

    for test in tests:
        test.generate_test()


if "__main__" == __name__:
    main()
