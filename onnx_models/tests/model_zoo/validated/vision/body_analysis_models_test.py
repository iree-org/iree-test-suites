# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://github.com/onnx/models/tree/main/validated/vision/body_analysis/

import pytest

from .....utils import *

artifacts_subdir = "model_zoo/validated/vision/body_analysis"


def test_age_gender_age_googlenet(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/body_analysis/age_gender/models/age_googlenet.onnx",
        artifacts_subdir=artifacts_subdir,
    )


def test_age_gender_gender_googlenet(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/body_analysis/age_gender/models/gender_googlenet.onnx",
        artifacts_subdir=artifacts_subdir,
    )


@pytest.mark.size_large
def test_age_gender_vgg_ilsvrc_16_age_imdb_wiki(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/body_analysis/age_gender/models/vgg_ilsvrc_16_age_imdb_wiki.onnx",
        artifacts_subdir=artifacts_subdir,
    )


@pytest.mark.size_large
def test_age_gender_vgg_ilsvrc_16_gender_imdb_wiki(
    compare_between_iree_and_onnxruntime,
):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/body_analysis/age_gender/models/vgg_ilsvrc_16_gender_imdb_wiki.onnx",
        artifacts_subdir=artifacts_subdir,
    )


def test_emotion_ferplus(
    compare_between_iree_and_onnxruntime,
):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        artifacts_subdir=artifacts_subdir,
    )


@pytest.mark.skip("ONNXRuntimeError")
def test_ultraface(
    compare_between_iree_and_onnxruntime,
):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx",
        artifacts_subdir=artifacts_subdir,
    )
