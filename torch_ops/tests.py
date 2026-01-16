import itertools
import numpy as np
import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    BlockMask,
    _LARGE_SPARSE_BLOCK_SIZE,
)
from generate import GenConfig, gen, gen_tests, test, Formula


def formulas(n, **kwargs):
    return tuple([Formula(**kwargs)] * n)


@gen_tests
class AB(torch.nn.Module):
    def forward(self, A, B):
        return A @ B

    @staticmethod
    @test
    def test_data():
        dynamic_shapes = {
            "A": {0: torch.export.Dim("N")},
            "B": {1: torch.export.Dim("N")},
        }
        args = formulas(2, shape=(64, 64))
        name = "Nx64xf32_64xNxf32"
        yield GenConfig(name, args=args, dynamic_shapes=dynamic_shapes, seed=0)

        args = formulas(2, shape=(64, 64), dtype=np.dtype("float16"))
        name = "Nx64xf16_64xNxf16"
        yield GenConfig(
            name, args=args, dynamic_shapes=dynamic_shapes, seed=1, rtol=1e-3, atol=1e-3
        )

    @staticmethod
    @test
    def benchmark_data():
        seed = 14
        SQUARE = [
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
        ]

        for n in SQUARE:
            name = f"{n}x{n}xf32_bench"
            args = formulas(2, shape=(n, n))
            yield GenConfig(name, args=args, seed=seed, mode="benchmark")
            seed += 1


@gen_tests
class AB_bfloat16(torch.nn.Module):
    def forward(self, A, B):
        A = A.to(torch.bfloat16)
        B = B.to(torch.bfloat16)
        return (A @ B).to(torch.float32)

    @staticmethod
    @test
    def test_data():
        args = formulas(2, shape=(64, 64))
        name = "Nx64xf32_64xNxf32"
        dynamic_shapes = {
            "A": {0: torch.export.Dim("N")},
            "B": {1: torch.export.Dim("N")},
        }
        yield GenConfig(name, args=args, dynamic_shapes=dynamic_shapes, seed=3)


@gen_tests
class ATB(torch.nn.Module):
    def forward(self, A, B):
        return A.t() @ B

    @staticmethod
    @test
    def test_data():
        args = formulas(2, shape=(64, 64))
        name = "64x64xf32"
        yield GenConfig(name, args=args, seed=4)

        args = formulas(2, shape=(64, 64), dtype=np.dtype("float16"))
        name = "64x64xf16"
        yield GenConfig(name, args=args, seed=5)


@gen_tests
class ABT(torch.nn.Module):
    def forward(self, A, B):
        return A @ B.t()

    @staticmethod
    @test
    def test_data():
        args = formulas(2, shape=(64, 64))
        name = "64x64xf32"
        yield GenConfig(name, args=args, seed=6)

        args = formulas(2, shape=(64, 64), dtype=np.dtype("float16"))
        name = "64x64xf16"
        yield GenConfig(name, args=args, seed=7, rtol=1e-3, atol=1e-3)


@gen_tests
class ABPlusC(torch.nn.Module):
    def forward(self, A, B, C):
        return A @ B + C

    @staticmethod
    @test
    def test_data():
        args = formulas(3, shape=(64, 64), coeff=2, offset=-1)
        name = "64x64xf32"
        yield GenConfig(name, args=args, seed=8, atol=1e-5)

        args = formulas(
            3, shape=(64, 64), coeff=2, offset=-1, dtype=np.dtype("float16")
        )
        name = "64x64xf16"
        yield GenConfig(name, args=args, seed=9, rtol=1e-3, atol=1e-3)


@gen_tests
class ReluABPlusC(torch.nn.Module):
    def forward(self, A, B, C):
        return torch.relu(A @ B + C)

    @staticmethod
    @test
    def test_data():
        args = formulas(3, shape=(64, 64), coeff=2, offset=-1)
        name = "64x64xf32"
        yield GenConfig(name, args=args, seed=10, atol=1e-5)

        args = formulas(
            3, shape=(64, 64), coeff=2, offset=-1, dtype=np.dtype("float16")
        )
        name = "64x64xf16"
        yield GenConfig(name, args=args, seed=11, rtol=1e-3, atol=1e-3)


@gen_tests
class GeluABPlusC(torch.nn.Module):
    def forward(self, A, B, C):
        return torch.ops.aten.gelu.default(A @ B + C)

    @staticmethod
    @test
    def test_data():
        args = formulas(3, shape=(64, 64), coeff=0.1, offset=-0.05)
        name = "64x64xf32"
        yield GenConfig(name, args=args, seed=12, atol=1e-5)

        args = formulas(3, shape=(64, 64), coeff=0.1, offset=-0.05)
        name = "64x64xf16"
        yield GenConfig(name, args=args, seed=13, rtol=1e-3, atol=1e-3)


@gen_tests
class FlexAttention(torch.nn.Module):
    def __init__(self):
        self._block_mask = self._create_empty_block_mask()
        super().__init__()

    def _create_empty_block_mask(self):
        return BlockMask.from_kv_blocks(
            kv_num_blocks=torch.ones([1, 1, 1], dtype=torch.int32),
            kv_indices=torch.zeros([1, 1, 1, 1], dtype=torch.int32),
            BLOCK_SIZE=_LARGE_SPARSE_BLOCK_SIZE,
            seq_lengths=(1, 1),
        )

    def score_mod_fn(self, score, batch, head, token_q, token_kv):
        return torch.tanh(score)

    def forward(self, Q, K, V):
        return flex_attention(
            Q,
            K,
            V,
            score_mod=self.score_mod_fn,
            block_mask=self._block_mask,
            scale=1.0,
            kernel_options={},
        )

    @staticmethod
    @test
    def test_data():
        args = formulas(3, shape=(4, 8, 1024, 64))
        name = "4x8x1024x64xf16"
        yield GenConfig(name, args=args, seed=14, rtol=1e-3, atol=1e-3)


@gen_tests
class InterestingShapesBiasAdd(torch.nn.Module):
    def forward(
        self,
        A,
        B,
        C,
        ACC,
        transposeA=False,
        transposeB=False,
        bias=False,
        castToBf16=False,
    ):
        if castToBf16:
            A = A.to(torch.bfloat16)
            B = B.to(torch.bfloat16)
            C = C.to(torch.bfloat16)
            ACC = ACC.to(torch.bfloat16)

        if transposeA:
            A = A.t()
        if transposeB:
            B = B.t()

        torch.matmul(A, B, out=ACC)

        if bias:
            ACC += C

        if castToBf16:
            ACC = ACC.to(torch.float32)

        return ACC

    @staticmethod
    @test
    def test_data():
        args = formulas(4, shape=(997, 997), dtype=np.dtype("float16"))
        kwargs = {"transposeA": False, "transposeB": True, "bias": True}
        name = "997x997xf16_NT_bias"
        yield GenConfig(name, args=args, kwargs=kwargs, seed=15, rtol=5e-3, atol=1e-3)

        args = formulas(4, shape=(997, 997), dtype=np.dtype("int8"))
        kwargs = {"transposeA": False, "transposeB": False, "bias": True}
        name = "997x997xi8_NN_bias"
        yield GenConfig(name, args=args, kwargs=kwargs, seed=16)

        args = formulas(4, shape=(997, 997))
        kwargs = {"transposeA": False, "transposeB": False, "bias": True}
        name = "997x997xf32_TN_bias"
        yield GenConfig(name, args=args, kwargs=kwargs, seed=17)

        A = Formula(shape=(3240000, 2))
        B = Formula(shape=(2, 2))
        C = Formula(shape=(3240000, 2))
        ACC = Formula(shape=(3240000, 2))
        args = (A, B, C, ACC)
        kwargs = {
            "transposeA": False,
            "transposeB": True,
            "bias": False,
            "castToBf16": True,
        }
        name = "3240000x2xbf16_matmul_2x2xbf16_NT"
        yield GenConfig(name, args=args, kwargs=kwargs, seed=18, rtol=1e-3, atol=1e-3)

        A = Formula(shape=(6144, 419))
        B = Formula(shape=(384, 419))
        C = Formula(shape=(6144, 384))
        ACC = Formula(shape=(6144, 384))
        args = (A, B, C, ACC)
        kwargs = {
            "transposeA": False,
            "transposeB": True,
            "bias": False,
            "castToBf16": True,
        }
        name = "6144x419xbf16_matmul_419x384xbf16_NT"
        yield GenConfig(name, args=args, kwargs=kwargs, seed=19, rtol=7e-3, atol=5e-3)

        A = Formula(shape=(1536, 64))
        B = Formula(shape=(35, 64))
        C = Formula(shape=(1536, 35))
        ACC = Formula(shape=(1536, 35))
        args = (A, B, C, ACC)
        kwargs = {"transposeA": False, "transposeB": True, "bias": False}
        name = "1536x64xbf16_matmul_64x35xbf16_NT"
        yield GenConfig(name, args=args, kwargs=kwargs, seed=20)

        A = Formula(shape=(1152, 997), dtype=np.dtype("float16"))
        B = Formula(shape=(997, 576), dtype=np.dtype("float16"))
        C = Formula(shape=(1152, 576), dtype=np.dtype("float16"))
        ACC = Formula(shape=(1152, 576), dtype=np.dtype("float16"))
        args = (A, B, C, ACC)
        kwargs = {"transposeA": False, "transposeB": False, "bias": False}
        name = "1152x997xf16_matmul_997x576xf16_NN"
        yield GenConfig(name, args=args, kwargs=kwargs, seed=20, rtol=1e-3, atol=1e-3)


# Just creating an instance generates tests.
AB()
AB_bfloat16()
ATB()
ABT()
ABPlusC()
ReluABPlusC()
GeluABPlusC()
FlexAttention()
InterestingShapesBiasAdd()

# Example of generation using the functional approach.
# gen(AB_bfloat16(), AB_bfloat16.test_data())
# gen(ATB(), ATB.test_data())
# gen(ABT(), ABT.test_data())
# gen(ABT(), ABT.test_data())
# gen(ABPlusC(), ABPlusC.test_data())
# gen(ReluABPlusC(), ReluABPlusC.test_data())
# gen(GeluABPlusC(), GeluABPlusC.test_data())
# gen(FlexAttention(), FlexAttention.test_data())
