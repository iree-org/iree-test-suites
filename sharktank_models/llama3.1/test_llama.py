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
def compile_llama_hip(llama_mlir):
    if "HIP_TARGET" not in os.environ:
        raise RuntimeError("HIP_TARGET not set")

    target_gpu = os.environ["HIP_TARGET"]
    return iree.compiler.compile_file(
        llama_mlir,
        extra_args=["--iree-hal-target-device=hip", f"--iree-hip-target={target_gpu}"],
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

        self._cache = self.allocate_cache()

        self._prefill = self.modules[-1].prefill_bs1
        self._decode = self.modules[-1].decode_bs1

    def prefill(self, *, ids):
        id_len = len(ids)

        # Pad out to the blocked sequence length
        pad = (block_size - id_len % block_size) % block_size
        ids = ids + [0] * pad
        padded_len = id_len + pad

        ids = numpy.asarray([ids], dtype=numpy.int64)
        length = numpy.asarray([id_len], numpy.int64)
        pages = padded_len // block_size
        pages = numpy.asarray([list(range(pages))], numpy.int64)

        logits = self._prefill(ids[:, :padded_len], length, pages, *self._cache)
        logits = numpy.asarray(logits).astype(numpy.float32)
        return logits[:, :id_len]

    def decode(self, *, ids, start=0):
        N = len(ids) + start
        pages = N // block_size + 1
        pages = numpy.asarray([list(range(pages))], numpy.int64)
        cache = self._cache

        all_logits = []
        for idx, id in enumerate(ids):
            id = numpy.asarray([[id]], dtype=numpy.int64)
            start_position = numpy.asarray([idx + start], dtype=numpy.int64)
            seq_len = numpy.asarray([idx + 1 + start], dtype=numpy.int64)

            logits = self._decode(id, seq_len, start_position, pages, *cache)
            all_logits.append(logits.to_host())

        logits = numpy.concatenate(all_logits, axis=1).astype(numpy.float32)
        return logits

    def allocate_cache(self):
        return [self.to_device(numpy.zeros((128, page_size), numpy.float16))]

    def to_device(self, a):
        return iree.runtime.asdevicearray(self.device, a)


@pytest.fixture
def toy_llama_cpu(compile_llama_cpu, llama_irpa):
    return ToyLlama(compiled=compile_llama_cpu, device_id="local-task", irpa=llama_irpa)


@pytest.fixture
def toy_llama_hip(compile_llama_hip, llama_irpa):
    return ToyLlama(compiled=compile_llama_hip, device_id="hip", irpa=llama_irpa)


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
    logits = toy_llama.prefill(ids=ids)
    ids = numpy.asarray([ids])

    logits = logits[0, :-1, :]
    ids = ids[:, 1:]

    return cross_entropy(logits, ids)


def decode_cross_entropy(toy_llama, ids):
    N = len(ids)
    logits = toy_llama.decode(ids=ids, start=0)
    ids = numpy.asarray([ids])

    # Compare predictions to selected tokens
    logits = logits[0, :-1, :]
    ids = ids[:, 1:]
    return cross_entropy(logits, ids)


def prefill_decode_cross_entropy(toy_llama, ids):
    N = len(ids)
    prefill_ids = ids[: N // 2]
    decode_ids = ids[N // 2 :]

    prefill_logits = toy_llama.prefill(ids=prefill_ids)
    decode_logits = toy_llama.decode(ids=decode_ids, start=len(prefill_ids))
    debug_logits = toy_llama.prefill(ids=ids)

    ids = numpy.asarray([ids])
    logits = numpy.concatenate([prefill_logits, decode_logits], axis=-2)

    # Compare predictions to selected tokens
    logits = logits[0, :-1, :]
    ids = ids[:, 1:]
    return cross_entropy(logits, ids)


@pytest.mark.target_cpu
def test_prefill_cpu(toy_llama_cpu):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    cross_entropy = prefill_cross_entropy(toy_llama_cpu, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-3
    ), "cross entropy outside of tolerance"


@pytest.mark.target_hip
def test_prefill_hip(toy_llama_hip):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    cross_entropy = prefill_cross_entropy(toy_llama_hip, ids)

    if cross_entropy is None:
        return

    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-3
    ), "cross entropy outside of tolerance"


@pytest.mark.target_cpu
def test_decode_cpu(toy_llama_cpu):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    cross_entropy = decode_cross_entropy(toy_llama_cpu, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-3
    ), "cross entropy outside of tolerance"


@pytest.mark.target_hip
def test_decode_hip(toy_llama_hip):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    cross_entropy = decode_cross_entropy(toy_llama_hip, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-3
    ), "cross entropy outside of tolerance"


@pytest.mark.target_cpu
def test_prefill_decode_cpu(toy_llama_cpu):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    cross_entropy = prefill_decode_cross_entropy(toy_llama_cpu, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-3
    ), "cross entropy outside of tolerance"


@pytest.mark.target_hip
def test_prefill_decode_hip(toy_llama_hip):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    cross_entropy = prefill_decode_cross_entropy(toy_llama_hip, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-3
    ), "cross entropy outside of tolerance"
