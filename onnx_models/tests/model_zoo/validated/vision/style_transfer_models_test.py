# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://github.com/onnx/models/tree/main/validated/vision/style_transfer/

import pytest

from .....utils import *

artifacts_subdir = "model_zoo/validated/vision/style_transfer"


def test_fast_neural_style(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx",
        artifacts_subdir=artifacts_subdir,
    )
