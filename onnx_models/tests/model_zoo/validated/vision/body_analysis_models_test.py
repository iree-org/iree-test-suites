# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://github.com/onnx/models/tree/main/validated/vision/body_analysis/

import pytest

from .....utils import *

ARTIFACTS_SUBDIR = "model_zoo/validated/vision/body_analysis"
BASE_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"


@pytest.mark.parametrize(
    "model",
    [
        # fmt: off
        pytest.param("age_gender/models/age_googlenet.onnx"),
        pytest.param("age_gender/models/gender_googlenet.onnx"),
        pytest.param("age_gender/models/vgg_ilsvrc_16_age_imdb_wiki.onnx", marks=pytest.mark.size_large),
        pytest.param("age_gender/models/vgg_ilsvrc_16_gender_imdb_wiki.onnx", marks=pytest.mark.size_large),
        pytest.param("emotion_ferplus/model/emotion-ferplus-8.onnx"),
        pytest.param("ultraface/models/version-RFB-320.onnx", marks=pytest.mark.skip("ONNXRuntimeError")),
        # fmt: on
    ],
)
def test_models(compare_between_iree_and_onnxruntime, model):
    compare_between_iree_and_onnxruntime(
        model_url=BASE_URL + model, artifacts_subdir=ARTIFACTS_SUBDIR
    )
