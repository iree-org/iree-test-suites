# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from .utils import *

# Note: can mark tests as expected to fail at a specific stage with:
# @pytest.mark.xfail(raises=IreeCompileException)
# @pytest.mark.xfail(raises=IreeRunException)


# https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
def test_mnist_7(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-7.onnx",
    )


# https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet
@pytest.mark.xfail(raises=IreeRunException)
def test_mobilenetv2_12(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
    )


# https://github.com/onnx/models/tree/main/validated/vision/classification/resnet
def test_resnet50_v1_12(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx",
    )


# https://github.com/onnx/models/tree/main/validated/vision/classification/alexnet
def test_alexnet_9(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/alexnet/model/bvlcalexnet-9.onnx",
    )
