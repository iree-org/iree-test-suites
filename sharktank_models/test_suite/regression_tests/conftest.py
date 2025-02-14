# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

class VmfbManager:
    cpu_vmfb = None
    rocm_vmfb = None
    pipeline_cpu_vmfb = None
    pipeline_rocm_vmfb = None

def pytest_addoption(parser):
    parser.addoption(
        "--model-name", 
        action="store", 
        type=str, 
        default=""
    )
    parser.addoption(
        "--submodel-name", 
        action="store", 
        type=str, 
        default=""
    )