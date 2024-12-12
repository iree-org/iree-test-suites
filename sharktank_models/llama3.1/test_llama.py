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

block_size = 32
page_size = 768 * block_size

THIS_DIR = pathlib.Path(__file__).parent
llama_mlir = str(THIS_DIR / "assets/toy_llama.mlir")
llama_irpa = [str(THIS_DIR / "assets/toy_llama.irpa")]

llama_tp2_mlir = str(THIS_DIR / "assets/toy_llama_tp2.mlir")
llama_tp2_irpa = [
    str(THIS_DIR / "assets/toy_llama_tp2.irpa"),
    str(THIS_DIR / "assets/toy_llama_tp2.rank0.irpa"),
    str(THIS_DIR / "assets/toy_llama_tp2.rank1.irpa"),
]


class ToyLlama:
    def __init__(self, compiled, device_id, irpa, sharding=1):
        """ToyLlama is wrapper / initalizer for an exported llama model

        Args:
            Compiled:  the compiled VMFB
            Device_id: the device id of the target device
            Irps:      the parameter weight file
            Sharding:  the number of devices used by model
        """
        paramIndex = iree.runtime.ParameterIndex()
        for i in irpa:
            paramIndex.load(i)
        provider = paramIndex.create_provider("model")

        self.sharding = sharding
        self.instance = iree.runtime.VmInstance()
        self.devices = [
            iree.runtime.get_device(device_id) for _ in range(self.sharding)
        ]
        self.config = iree.runtime.Config(device=self.devices[0])
        self.parameters = iree.runtime.create_io_parameters_module(
            self.instance, provider
        )
        self.hal = iree.runtime.create_hal_module(self.instance, devices=self.devices)
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
        return [
            self.to_device(
                numpy.zeros((128, page_size // self.sharding), numpy.float16), d
            )
            for d in self.devices
        ]

    def to_device(self, a, d):
        return iree.runtime.asdevicearray(d, a)


def cpu_flags(sharding):
    return ["--iree-hal-target-device=llvm-cpu"] * sharding + [
        "--iree-llvmcpu-target-cpu=host"
    ]


def hip_flags(sharding):
    if "HIP_TARGET" not in os.environ:
        raise RuntimeError("HIP_TARGET not set")

    target_gpu = os.environ["HIP_TARGET"]
    return [f"--iree-hal-target-device=hip[{i}]" for i in range(sharding)] + [
        f"--iree-hip-target={target_gpu}",
        # TODO: Remove once https://github.com/iree-org/iree/issues/19347 is addressed
        "--iree-codegen-block-dynamic-dimensions-of-contractions=false",
    ]


CPU_TP1_CONFIG = (cpu_flags, llama_mlir, "local-task", llama_irpa, 1)
HIP_TP1_CONFIG = (hip_flags, llama_mlir, "hip", llama_irpa, 1)

CPU_TP2_CONFIG = (cpu_flags, llama_tp2_mlir, "local-task", llama_tp2_irpa, 2)
HIP_TP2_CONFIG = (hip_flags, llama_tp2_mlir, "hip", llama_tp2_irpa, 2)


@pytest.fixture(
    params=[
        pytest.param(CPU_TP1_CONFIG, marks=pytest.mark.target_cpu),
        pytest.param(CPU_TP2_CONFIG, marks=pytest.mark.target_cpu),
        pytest.param(HIP_TP1_CONFIG, marks=pytest.mark.target_hip),
        pytest.param(HIP_TP2_CONFIG, marks=pytest.mark.target_hip),
    ],
    ids=["cpu", "cpu_tp2", "hip", "hip_tp2"],
    scope="session",
)
def toy_llama(request):
    flags, mlir, device, irpa, sharding = request.param
    flags = flags(sharding)
    compiled = iree.compiler.compile_file(mlir, extra_args=flags)

    return ToyLlama(compiled=compiled, device_id=device, irpa=irpa, sharding=sharding)


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


def prefill_cross_entropy(model, ids):
    logits = model.prefill(ids=ids)
    ids = numpy.asarray([ids])

    logits = logits[0, :-1, :]
    ids = ids[:, 1:]

    return cross_entropy(logits, ids)


def decode_cross_entropy(model, ids):
    N = len(ids)
    logits = model.decode(ids=ids, start=0)
    ids = numpy.asarray([ids])

    # Compare predictions to selected tokens
    logits = logits[0, :-1, :]
    ids = ids[:, 1:]
    return cross_entropy(logits, ids)


def prefill_decode_cross_entropy(model, ids):
    N = len(ids)
    prefill_ids = ids[: N // 2]
    decode_ids = ids[N // 2 :]

    prefill_logits = model.prefill(ids=prefill_ids)
    decode_logits = model.decode(ids=decode_ids, start=len(prefill_ids))

    ids = numpy.asarray([ids])
    logits = numpy.concatenate([prefill_logits, decode_logits], axis=-2)

    # Compare predictions to selected tokens
    logits = logits[0, :-1, :]
    ids = ids[:, 1:]
    return cross_entropy(logits, ids)


def test_prefill(toy_llama):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    cross_entropy = prefill_cross_entropy(toy_llama, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-2
    ), "cross entropy outside of tolerance"


def test_decode(toy_llama):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    cross_entropy = decode_cross_entropy(toy_llama, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-2
    ), "cross entropy outside of tolerance"


def test_prefill_decode(toy_llama):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    cross_entropy = prefill_decode_cross_entropy(toy_llama, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-1
    ), "cross entropy outside of tolerance"
