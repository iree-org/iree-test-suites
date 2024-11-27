import iree.compiler
import iree.runtime
import numpy
import os
import pytest
import torch

page_size = 12288
block_size = 16


@pytest.fixture
def llama_mlir():
    dirname = os.path.dirname(__file__)
    return f"{dirname}/assets/toy_llama.mlir"


@pytest.fixture
def llama_irpa():
    dirname = os.path.dirname(__file__)
    return f"{dirname}/assets/toy_llama.irpa"


@pytest.fixture
def compile_llama_cpu(llama_mlir):
    return iree.compiler.compile_file(
        llama_mlir,
        extra_args=[
            "--iree-hal-target-device=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
        ],
    )


@pytest.fixture
def compile_llama_gfx1100(llama_mlir):
    return iree.compiler.compile_file(
        llama_mlir,
        extra_args=["--iree-hal-target-device=hip", "--iree-hip-target=gfx1100"],
    )


class ToyLlama:
    def __init__(self, compiled, device_id, irpa):
        paramIndex = iree.runtime.ParameterIndex()
        paramIndex.load(irpa)
        provider = paramIndex.create_provider("model")

        self.instance = iree.runtime.VmInstance()
        self.device = iree.runtime.get_device(device_id)
        self.config = iree.runtime.Config(device=self.device)
        self.parameters = iree.runtime.create_io_parameters_module(
            self.instance, provider
        )
        self.hal = iree.runtime.create_hal_module(self.instance, self.device)
        self.binary = iree.runtime.VmModule.copy_buffer(self.instance, compiled)
        self.modules = iree.runtime.load_vm_modules(
            self.parameters, self.hal, self.binary, config=self.config
        )

        self.prefill = self.modules[-1].prefill_bs1
        self.decode = self.modules[-1].decode_bs1


@pytest.fixture
def toy_llama_cpu(compile_llama_cpu, llama_irpa):
    return ToyLlama(compile_llama_cpu, "local-sync", llama_irpa)


@pytest.fixture
def toy_llama_gfx1100(compile_llama_gfx1100, llama_irpa):
    return ToyLlama(compile_llama_gfx1100, "hip", llama_irpa)


def prefill_cross_entropy(toy_llama, ids):
    assert ids.shape[1] % block_size == 0

    len = numpy.asarray([ids.shape[1]], numpy.int64)
    pages = ids.shape[1] // block_size
    pages = numpy.asarray([list(range(pages))], numpy.int64)
    cache = numpy.zeros((128, page_size), numpy.float16)

    logits = toy_llama.prefill(ids[:, : ids.shape[1]], len, pages, cache)

    logits = torch.Tensor(numpy.asarray(logits)).to(torch.float32)
    ids = torch.Tensor(numpy.asarray(ids)).to(torch.int64)

    logits = logits[0, :-1, :]
    ids = ids[0, 1 : ids.shape[1]]

    return torch.nn.functional.cross_entropy(logits, ids, ignore_index=0)


def test_prefill_cpu(toy_llama_cpu):
    ids = numpy.array(
        [[0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]],
        numpy.int64,
    )
    cross_entropy = prefill_cross_entropy(toy_llama_cpu, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(0.589, 1e-3)


def test_prefill_gfx1100(toy_llama_gfx1100):
    ids = numpy.array(
        [[0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]],
        numpy.int64,
    )
    cross_entropy = prefill_cross_entropy(toy_llama_gfx1100, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(0.589, 1e-3)
