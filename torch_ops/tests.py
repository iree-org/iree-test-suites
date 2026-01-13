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


# Just creating an instance generates tests.
AB()
AB_bfloat16()
ATB()
ABT()
ABPlusC()
ReluABPlusC()
GeluABPlusC()
FlexAttention()

# Example of generation using the functional approach.
# gen(AB_bfloat16(), AB_bfloat16.test_data())
# gen(ATB(), ATB.test_data())
# gen(ABT(), ABT.test_data())
# gen(ABT(), ABT.test_data())
# gen(ABPlusC(), ABPlusC.test_data())
# gen(ReluABPlusC(), ReluABPlusC.test_data())
# gen(GeluABPlusC(), GeluABPlusC.test_data())
# gen(FlexAttention(), FlexAttention.test_data())
