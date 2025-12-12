# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://huggingface.co/onnxmodelzoo/legacy_models/tree/main/validated/vision/style_transfer

import pytest

from .....utils import *

BASE_PATH = "validated/vision/style_transfer/"


@pytest.mark.parametrize(
    "model",
    [
        # fmt: off
        pytest.param("fast_neural_style/model/mosaic-9.onnx"),
        # fmt: on
    ],
)
def test_models(compare_between_iree_and_onnxruntime, model):
    compare_between_iree_and_onnxruntime(
        model_url=BASE_PATH + model, artifacts_subdir=BASE_PATH
    )
