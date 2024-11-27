# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.compiler
import iree.runtime
import numpy
import os
import pathlib
import pytest

page_size = 12288
block_size = 16

THIS_DIR = pathlib.Path(__file__).parent


@pytest.fixture
def llama_mlir():
    return str(THIS_DIR / "assets/toy_llama.mlir")


@pytest.fixture
def llama_irpa():
    return str(THIS_DIR / "assets/toy_llama.irpa")


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
    if "HIP_TARGET" not in os.environ:
        return None

    target_gpu = os.environ["HIP_TARGET"]
    return iree.compiler.compile_file(
        llama_mlir,
        extra_args=["--iree-hal-target-device=hip", f"--iree-hip-target={target_gpu}"],
    )


class ToyLlama:
    def __init__(self, compiled, device_id, irpa):
        if compiled is None:
            return

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
    return ToyLlama(compiled=compile_llama_cpu, device_id="local-task", irpa=llama_irpa)


@pytest.fixture
def toy_llama_gfx1100(compile_llama_gfx1100, llama_irpa):
    return ToyLlama(compiled=compile_llama_gfx1100, device_id="hip", irpa=llama_irpa)


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    # predictions = numpy.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]

    predictions = numpy.exp(predictions)
    selected = predictions[numpy.arange(N), targets]
    sum = numpy.sum(predictions, axis=-1)
    norm = selected / sum
    norm = -numpy.log(norm)
    ce = numpy.sum(norm) / N
    return ce


def prefill_cross_entropy(toy_llama, ids):
    if not hasattr(toy_llama, "prefill"):
        return

    assert ids.shape[1] % block_size == 0

    len = numpy.asarray([ids.shape[1]], numpy.int64)
    pages = ids.shape[1] // block_size
    pages = numpy.asarray([list(range(pages))], numpy.int64)
    cache = numpy.zeros((128, page_size), numpy.float16)

    logits = toy_llama.prefill(ids[:, : ids.shape[1]], len, pages, cache)

    logits = numpy.asarray(logits).astype(numpy.float32)
    ids = numpy.asarray(ids)

    logits = logits[0, :-1, :]
    ids = ids[0, 1 : ids.shape[1]]

    return cross_entropy(logits, ids)


@pytest.mark.target_cpu
def test_prefill_cpu(toy_llama_cpu):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = numpy.array(
        [[0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]],
        numpy.int64,
    )
    cross_entropy = prefill_cross_entropy(toy_llama_cpu, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-3
    ), "cross entropy outside of tolerance"


@pytest.mark.target_hip
def test_prefill_gfx1100(toy_llama_gfx1100):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = numpy.array(
        [[0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]],
        numpy.int64,
    )
    cross_entropy = prefill_cross_entropy(toy_llama_gfx1100, ids)

    if cross_entropy is None:
        return

    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-3
    ), "cross entropy outside of tolerance"
