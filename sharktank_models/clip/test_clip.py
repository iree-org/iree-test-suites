# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Union
import pytest
import pathlib
from os import PathLike
import os
import numpy as np
import ml_dtypes

import iree.runtime
import iree.compiler

THIS_DIR = pathlib.Path(__file__).parent
ASSET_PATH = os.getenv("ASSET_PATH", default=str(THIS_DIR)) + "/clip"

def load_tensor_from_irpa(path: PathLike) -> np.ndarray:
    index = iree.runtime.ParameterIndex()
    index.load(str(path))
    index_entry: iree.runtime.ParameterIndexEntry = index.items()[0][1]
    return iree.runtime.parameter_index_entry_as_numpy_ndarray(index_entry)


@pytest.fixture(
    params=[
        pytest.param("local-task", marks=pytest.mark.target_cpu),
        pytest.param("hip", marks=pytest.mark.target_hip),
    ]
)
def device_id(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=["bf16", "f32"])
def model_variant(request: pytest.FixtureRequest) -> str:
    return request.param


mlir_path = {
    "bf16": f"{ASSET_PATH}/assets/text_model/toy/bf16.mlir",
    "f32": f"{ASSET_PATH}/assets/text_model/toy/f32.mlir",
}

parameters_path = {
    "bf16": f"{ASSET_PATH}/assets/text_model/toy/bf16_parameters.irpa",
    "f32": f"{ASSET_PATH}/assets/text_model/toy/f32_parameters.irpa",
}

function_arg0_path = f"{ASSET_PATH}/assets/text_model/toy/forward_bs4_arg0_input_ids.irpa"
function_expected_result0 = (
    f"{ASSET_PATH}/assets/text_model/toy/forward_bs4_expected_result0_last_hidden_state_f32.irpa"
)

absolute_tolerance = {
    "bf16": 1e-3,
    "f32": 1e-5,
}


def compiler_args(device_id: str) -> list[str]:
    if device_id == "local-task":
        return ["--iree-hal-target-device=llvm-cpu", "--iree-llvmcpu-target-cpu=host"]
    if device_id == "hip":
        if "HIP_TARGET" not in os.environ:
            raise RuntimeError("HIP_TARGET environment variable not set")

        hip_target = os.environ["HIP_TARGET"]
        return ["--iree-hal-target-device=hip", f"--iree-hip-target={hip_target}"]

    raise KeyError(f"Compiler args for {device_id} not found")


def compile_and_run(
    mlir_path: str, compiler_args: list[str], function: str, args: list[np.ndarray]
) -> list[np.ndarray]:
    iree.compiler.compile_file(
        mlir_path,
        extra_args=compiler_args,
    )


@pytest.fixture(scope="session")
def iree_module(model_variant, device_id) -> iree.runtime.VmModule:
    compiler_arguments = compiler_args(device_id)


def device_array_to_host(device_array: iree.runtime.DeviceArray) -> np.ndarray:
    def reinterpret_hal_buffer_view_element_type(
        buffer_view: iree.runtime.HalBufferView,
        element_type: iree.runtime.HalElementType,
    ) -> iree.runtime.HalBufferView:
        return iree.runtime.HalBufferView(
            buffer=buffer_view.get_buffer(),
            shape=buffer_view.shape,
            element_type=element_type,
        )

    def reinterpret_device_array_dtype(
        device_array: iree.runtime.DeviceArray, dtype: np.dtype
    ) -> iree.runtime.DeviceArray:
        return iree.runtime.DeviceArray(
            device=device_array._device,
            buffer_view=reinterpret_hal_buffer_view_element_type(
                device_array._buffer_view,
                iree.runtime.array_interop.map_dtype_to_element_type(dtype),
            ),
        )

    def bfloat16_device_array_to_numpy(
        device_array: iree.runtime.DeviceArray,
    ) -> np.ndarray:
        device_array_as_int16 = reinterpret_device_array_dtype(device_array, np.int16)
        np_array_as_int16 = device_array_as_int16.to_host()
        return np_array_as_int16.view(dtype=ml_dtypes.bfloat16)

    if device_array._buffer_view.element_type == int(
        iree.runtime.HalElementType.BFLOAT_16
    ):
        return bfloat16_device_array_to_numpy(device_array)
    else:
        return device_array.to_host()


def cosine_similarity(
    a: np.ndarray, b: np.ndarray, /, *, dim: Optional[Union[int, tuple[int]]] = None
) -> np.ndarray:
    """Compute cosine similarity over dimensions dim.
    If dim is none computes over all dimensions."""
    dot_product = np.sum(a * b, axis=dim)
    norm_a = np.sqrt(np.power(a, 2).sum(axis=dim))
    norm_b = np.sqrt(np.power(b, 2).sum(axis=dim))
    return dot_product / (norm_a * norm_b)


def assert_text_encoder_state_close(
    actual: np.ndarray, expected: np.ndarray, atol: float
):
    """The cosine similarity has been suggested to compare encoder states.

    Dehua Peng, Zhipeng Gui, Huayi Wu -
    Interpreting the Curse of Dimensionality from Distance Concentration and Manifold
    Effect (2023)

    shows that cosine and all Minkowski distances suffer from the curse of
    dimensionality.
    The cosine similarity ignores the vector magnitudes. We can probably come up with a
    better metric, but this is maybe good enough.

    The functions expects that the last dimension is the features per token.
    It will compute the cosine similarity for each token.
    """
    cosine_similarity_per_token = cosine_similarity(
        actual,
        expected,
        dim=-1,
    )
    np.testing.assert_allclose(
        cosine_similarity_per_token,
        np.ones_like(cosine_similarity_per_token),
        atol=atol,
        rtol=0,
    )


def test_results_close(model_variant, device_id):
    module_buffer = iree.compiler.compile_file(
        str(mlir_path[model_variant]),
        extra_args=compiler_args(device_id),
    )

    vm_instance = iree.runtime.VmInstance()
    paramIndex = iree.runtime.ParameterIndex()
    paramIndex.load(str(parameters_path[model_variant]))
    parameter_provider = paramIndex.create_provider("model")
    parameters_module = iree.runtime.create_io_parameters_module(
        vm_instance, parameter_provider
    )
    device = iree.runtime.get_device(device_id)
    hal_module = iree.runtime.create_hal_module(instance=vm_instance, devices=[device])
    vm_module = iree.runtime.VmModule.from_buffer(vm_instance, module_buffer)
    config = iree.runtime.Config(device=device)
    bound_modules = iree.runtime.load_vm_modules(
        hal_module, parameters_module, vm_module, config=config
    )
    module = bound_modules[-1]
    result = module.forward_bs4(load_tensor_from_irpa(function_arg0_path))[0]

    expected_result = load_tensor_from_irpa(function_expected_result0)
    result = device_array_to_host(result).astype(dtype=expected_result.dtype)

    assert_text_encoder_state_close(
        result, expected_result, absolute_tolerance[model_variant]
    )
