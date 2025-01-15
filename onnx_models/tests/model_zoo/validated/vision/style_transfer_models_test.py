# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://github.com/onnx/models/tree/main/validated/vision/style_transfer/

import pytest

from .....utils import *

ARTIFACTS_SUBDIR = "model_zoo/validated/vision/style_transfer"
BASE_URL = "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/"


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
        model_url=BASE_URL + model, artifacts_subdir=ARTIFACTS_SUBDIR
    )
