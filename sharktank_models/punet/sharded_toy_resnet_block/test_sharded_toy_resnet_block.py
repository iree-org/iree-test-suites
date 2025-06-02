# Copyright 2025 The IREE Authors
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

shard_count = 2


def load_tensors_from_irpa(path: PathLike) -> list[np.ndarray]:
    index = iree.runtime.ParameterIndex()
    index.load(str(path))
    index_dict = dict(index.items())
    index_dict = {k: v for k, v in index_dict.items() if k.isdigit()}
    return [
        iree.runtime.parameter_index_entry_as_numpy_ndarray(index_dict[f"{i}"])
        for i in range(len(index_dict.items()))
    ]


@pytest.fixture(
    params=[
        pytest.param("local-task", marks=pytest.mark.target_cpu),
        pytest.param("hip", marks=pytest.mark.target_hip),
    ]
)
def device_id(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=["f32"])
def model_variant(request: pytest.FixtureRequest) -> str:
    return request.param


def get_devices(driver: str) -> list[iree.runtime.HalDevice]:
    if driver == "local-task":
        return [iree.runtime.get_device(driver)] * shard_count

    # Get as many devices as are available if they are less then required,
    # repeat them to get the required count.
    hal_driver = iree.runtime.get_driver(driver)
    available_device_count = len(hal_driver.query_available_devices())
    device_ids = [f"{driver}://{i%available_device_count}" for i in range(shard_count)]
    return [iree.runtime.get_device(d) for d in device_ids]


mlir_path = {
    "f32": THIS_DIR / "assets/model.mlir",
}

parameters_path = {
    "f32": [THIS_DIR / f"assets/model.rank{i}.irpa" for i in range(shard_count)],
}

input_args_path = {
    "f32": THIS_DIR / "assets/input_args.irpa",
}

expected_results_path = {
    "f32": THIS_DIR / "assets/expected_results.irpa",
}

absolute_tolerance = {
    "f32": 5e-5,
}


def compiler_args(device_id: str) -> list[str]:
    res = []
    if device_id == "local-task":
        iree_hal_target_device = "llvm-cpu"
        res += ["--iree-llvmcpu-target-cpu=host"]
    elif device_id == "hip":
        if "HIP_TARGET" not in os.environ:
            raise RuntimeError("HIP_TARGET environment variable not set")

        hip_target = os.environ["HIP_TARGET"]
        iree_hal_target_device = "hip"
        res += [f"--iree-hip-target={hip_target}"]
    else:
        raise KeyError(f"Compiler args for {device_id} not found")

    res += [
        f"--iree-hal-target-device={iree_hal_target_device}[{i}]"
        for i in range(shard_count)
    ]
    return res


def test_results_close(model_variant, device_id):
    module_buffer = iree.compiler.compile_file(
        str(mlir_path[model_variant]),
        extra_args=compiler_args(device_id),
    )

    vm_instance = iree.runtime.VmInstance()
    paramIndex = iree.runtime.ParameterIndex()
    for path in parameters_path[model_variant]:
        paramIndex.load(str(path))
    parameter_provider = paramIndex.create_provider("model")
    parameters_module = iree.runtime.create_io_parameters_module(
        vm_instance, parameter_provider
    )
    devices = get_devices(device_id)
    hal_module = iree.runtime.create_hal_module(instance=vm_instance, devices=devices)
    vm_module = iree.runtime.VmModule.from_buffer(vm_instance, module_buffer)
    config = iree.runtime.Config(device=devices[0])
    bound_modules = iree.runtime.load_vm_modules(
        hal_module, parameters_module, vm_module, config=config
    )
    module = bound_modules[-1]
    results = module.main(*load_tensors_from_irpa(input_args_path[model_variant]))

    expected_results = load_tensors_from_irpa(expected_results_path[model_variant])
    actual_results = [
        results[i].to_host().astype(dtype=expected_results[i].dtype)
        for i in range(len(expected_results))
    ]

    np.testing.assert_allclose(
        actual_results, expected_results, atol=absolute_tolerance[model_variant], rtol=0
    )
