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

kv_size = 768
block_size = 32
page_size = kv_size * block_size

THIS_DIR = pathlib.Path(__file__).parent
ASSET_PATH = os.getenv("ASSET_PATH", default=str(THIS_DIR)) + "/llama3.1"

BS1_DIR =  f"{ASSET_PATH}/assets/bs1"
BS4_DIR = f"{ASSET_PATH}/assets/bs4"
BS32_DIR = f"{ASSET_PATH}/assets/bs32"

llama_mlir_bs1 = f"{BS1_DIR}/toy_llama_bs1.mlir"
llama_mlir_bs4 = f"{BS4_DIR}/toy_llama_bs4.mlir"
llama_mlir_bs32 = f"{BS32_DIR}/toy_llama_bs32.mlir"

# irpa files.
# this is a list because sharding would have multiple irpa files
llama_irpa = [f"{ASSET_PATH}/assets/toy_llama.irpa"]


class ToyLlama:
    def __init__(self, compiled, device_id, irpa, batch_size=4, sharding=1):
        """ToyLlama is wrapper / initalizer for an exported llama model

        Args:
            Compiled:   the compiled VMFB
            Device_id:  the device id of the target device
            Irpa:       the parameter weight file
            Batch_size: the batch size for this model
            Sharding:   the number of devices used by model
        """
        paramIndex = iree.runtime.ParameterIndex()
        for i in irpa:
            paramIndex.load(i)
        provider = paramIndex.create_provider("model")

        self.batch_size = batch_size
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

        # Select the appropriate functions based on batch size
        self._prefill = getattr(self.modules[-1], f"prefill_bs{batch_size}")
        self._decode = getattr(self.modules[-1], f"decode_bs{batch_size}")
        self.use_native_batch = True
        print(f"Using native batch size {batch_size} functions")

    def prefill(self, *, ids):
        id_len = len(ids)

        # Pad out to the blocked sequence length
        pad = (block_size - id_len % block_size) % block_size
        ids = ids + [0] * pad
        padded_len = id_len + pad

        # Create a batch of identical inputs for testing
        batched_ids = numpy.tile(
            numpy.asarray([ids], dtype=numpy.int64), (self.batch_size, 1)
        )
        length = numpy.full([self.batch_size], id_len, numpy.int64)
        pages = padded_len // block_size
        # Each batch item gets the same pages
        pages = numpy.tile(
            numpy.asarray([list(range(pages))], numpy.int64), (self.batch_size, 1)
        )

        if self.use_native_batch:
            # Use native batch function - one call for the entire batch
            logits = self._prefill(batched_ids, length, pages, *self._cache)
            # Convert device array to host
            logits = logits.to_host().astype(numpy.float32)
            return logits[:, :id_len]
        else:
            # Process each batch item separately since we're using bs1 functions
            all_logits = []
            for i in range(self.batch_size):
                # Use the bs1 function for each batch item
                logits = self._prefill(
                    batched_ids[i : i + 1],
                    length[i : i + 1],
                    pages[i : i + 1],
                    *self._cache,
                )
                # Convert to host array before appending
                all_logits.append(logits.to_host())

            # Combine results from all batch items
            logits = numpy.concatenate(all_logits, axis=0).astype(numpy.float32)
            return logits[:, :id_len]

    def decode(self, *, ids, start=0):
        N = len(ids) + start
        pages = N // block_size + 1
        base_pages = numpy.asarray([list(range(pages))], numpy.int64)
        batched_pages = numpy.tile(base_pages, (self.batch_size, 1))
        cache = self._cache

        if self.use_native_batch:
            # Process tokens one at a time using the batch-specific functions
            all_logits = []
            for idx, id_val in enumerate(ids):
                # Create a batch of the same token for all batch elements
                # Shape is batch_size x 1 as expected by decode_bsX functions
                batched_id = numpy.full([self.batch_size, 1], id_val, dtype=numpy.int64)
                batched_seq_len = numpy.full(
                    [self.batch_size], idx + 1 + start, dtype=numpy.int64
                )
                batched_start_position = numpy.full(
                    [self.batch_size], idx + start, dtype=numpy.int64
                )

                logits = self._decode(
                    batched_id,
                    batched_seq_len,
                    batched_start_position,
                    batched_pages,
                    *cache,
                )
                all_logits.append(logits.to_host())

            # Combine results along sequence dimension
            if len(all_logits) > 1:
                combined = numpy.concatenate(all_logits, axis=1).astype(numpy.float32)
                return combined
            else:
                # Just one token's worth of logits
                return all_logits[0].astype(numpy.float32)
        else:
            # Fallback for models without native batch support
            # Process one batch item at a time using bs1 functions
            batch_logits = []
            for idx, id in enumerate(ids):
                id_array = numpy.asarray([[id]], dtype=numpy.int64)
                start_position = numpy.asarray([idx + start], dtype=numpy.int64)
                seq_len = numpy.asarray([idx + 1 + start], dtype=numpy.int64)

                logits = self._decode(
                    id_array, seq_len, start_position, base_pages, *cache
                )
                batch_logits.append(logits.to_host())

            # Combine results for the token sequence
            combined = numpy.concatenate(batch_logits, axis=1).astype(numpy.float32)

            # Duplicate for all batch items (simpler approach for MVP)
            return numpy.tile(combined, (self.batch_size, 1, 1))

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
    return [f"--iree-hal-target-device=llvm-cpu[{i}]" for i in range(sharding)] + [
        "--iree-llvmcpu-target-cpu=host"
    ]


def hip_flags(sharding):
    if "HIP_TARGET" not in os.environ:
        raise RuntimeError("HIP_TARGET not set")

    target_gpu = os.environ["HIP_TARGET"]
    return [f"--iree-hal-target-device=hip[{i}]" for i in range(sharding)] + [
        f"--iree-hip-target={target_gpu}",
    ]


# Define configs for each batch size
CPU_BS1_CONFIG = (cpu_flags, llama_mlir_bs1, "local-task", llama_irpa, 1, 1)
CPU_BS4_CONFIG = (cpu_flags, llama_mlir_bs4, "local-task", llama_irpa, 4, 1)
CPU_BS32_CONFIG = (cpu_flags, llama_mlir_bs32, "local-task", llama_irpa, 32, 1)


@pytest.fixture(
    params=[
        pytest.param(CPU_BS1_CONFIG, marks=pytest.mark.target_cpu),
        pytest.param(CPU_BS4_CONFIG, marks=pytest.mark.target_cpu),
        pytest.param(CPU_BS32_CONFIG, marks=pytest.mark.target_cpu),
    ],
    ids=["cpu_bs1", "cpu_bs4", "cpu_bs32"],
    scope="session",
)
def toy_llama_batched(request):
    flags, mlir, device, irpa, batch_size, sharding = request.param
    flags = flags(sharding)
    compiled = iree.compiler.compile_file(mlir, extra_args=flags)

    return ToyLlama(
        compiled=compiled,
        device_id=device,
        irpa=irpa,
        batch_size=batch_size,
        sharding=sharding,
    )


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

    # Test the first batch item (all should be identical)
    logits_item = logits[0, :-1, :]
    ids_array = numpy.asarray([ids])
    ids_item = ids_array[:, 1:]

    return cross_entropy(logits_item, ids_item)


def decode_cross_entropy(model, ids):
    N = len(ids)
    logits = model.decode(ids=ids, start=0)
    ids_array = numpy.asarray([ids])

    # Compare predictions to selected tokens using first batch item
    logits_item = logits[0, :-1, :]
    ids_item = ids_array[:, 1:]
    return cross_entropy(logits_item, ids_item)


def prefill_decode_cross_entropy(model, ids):
    N = len(ids)
    prefill_ids = ids[: N // 2]
    decode_ids = ids[N // 2 :]

    prefill_logits = model.prefill(ids=prefill_ids)
    decode_logits = model.decode(ids=decode_ids, start=len(prefill_ids))

    ids_array = numpy.asarray([ids])
    # Use first batch item for evaluation
    logits = numpy.concatenate([prefill_logits[0:1], decode_logits[0:1]], axis=1)

    # Compare predictions to selected tokens
    logits_item = logits[0, :-1, :]
    ids_item = ids_array[:, 1:]
    return cross_entropy(logits_item, ids_item)


def test_prefill(toy_llama_batched):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    cross_entropy = prefill_cross_entropy(toy_llama_batched, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-1
    ), "cross entropy outside of tolerance"


def test_decode(toy_llama_batched):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    cross_entropy = decode_cross_entropy(toy_llama_batched, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-1
    ), "cross entropy outside of tolerance"


def test_prefill_decode(toy_llama_batched):
    # These are the maximized selected tokens when prompted with 0. It is designed to get the highest possible cross entropy.
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    cross_entropy = prefill_decode_cross_entropy(toy_llama_batched, ids)
    cross_entropy = cross_entropy.item()
    assert cross_entropy == pytest.approx(
        0.589, 1e-1
    ), "cross entropy outside of tolerance"


def test_batch_consistency(toy_llama_batched):
    """Test that all elements in a batch have the same outputs when given identical inputs."""
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    logits = toy_llama_batched.prefill(ids=ids)

    # Check that all batch outputs are identical to the first batch element
    for i in range(1, toy_llama_batched.batch_size):
        numpy.testing.assert_allclose(
            logits[0],
            logits[i],
            rtol=1e-2,
            atol=1e-2,
            err_msg=f"Batch element {i} differs from element 0",
        )

    # Also test decode consistency
    decode_ids = ids[:5]  # Use shorter sequence for decode
    decode_logits = toy_llama_batched.decode(ids=decode_ids, start=0)

    # Check decode batch consistency
    for i in range(1, toy_llama_batched.batch_size):
        numpy.testing.assert_allclose(
            decode_logits[0],
            decode_logits[i],
            rtol=1e-2,
            atol=1e-2,
            err_msg=f"Decode batch element {i} differs from element 0",
        )
