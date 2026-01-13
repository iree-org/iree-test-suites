# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://huggingface.co/onnxmodelzoo/legacy_models/tree/main/validated/vision/classification

import pytest

from .....utils import *

BASE_PATH = "validated/vision/classification/"


@pytest.mark.parametrize(
    "model",
    [
        # fmt: off
        pytest.param("alexnet/model/bvlcalexnet-12.onnx"),
        pytest.param("caffenet/model/caffenet-12.onnx"),
        pytest.param("densenet-121/model/densenet-12.onnx"),
        pytest.param("efficientnet-lite4/model/efficientnet-lite4-11.onnx"),
        pytest.param("inception_and_googlenet/googlenet/model/googlenet-12.onnx"),
        pytest.param("inception_and_googlenet/inception_v1/model/inception-v1-12.onnx"),
        pytest.param("inception_and_googlenet/inception_v2/model/inception-v2-9.onnx"),
        pytest.param("mnist/model/mnist-12.onnx"),
        pytest.param("mobilenet/model/mobilenetv2-12.onnx"),
        pytest.param("rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.onnx"),
        pytest.param("resnet/model/resnet50-v1-12.onnx"),
        pytest.param("resnet/model/resnet50-v2-7.onnx"),
        pytest.param("shufflenet/model/shufflenet-9.onnx"),
        pytest.param("shufflenet/model/shufflenet-v2-12.onnx"),
        pytest.param("squeezenet/model/squeezenet1.0-9.onnx"),
        pytest.param("vgg/model/vgg19-7.onnx", marks=pytest.mark.size_large),
        pytest.param("zfnet-512/model/zfnet512-12.onnx", marks=pytest.mark.size_large),
        # fmt: on
    ],
)
def test_models(compare_between_iree_and_onnxruntime, model):
    compare_between_iree_and_onnxruntime(
        model_url=BASE_PATH + model,
        artifacts_subdir=BASE_PATH,
    )
