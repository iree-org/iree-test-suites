# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://github.com/onnx/models/tree/main/validated/vision/classification/

import pytest

from ...utils import *


@pytest.mark.xfail(raises=IreeCompileException)
def test_alexnet(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/alexnet/model/bvlcalexnet-12.onnx",
        artifacts_subdir="vision/classification",
    )


def test_caffenet(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/caffenet/model/caffenet-12.onnx",
        artifacts_subdir="vision/classification",
    )


def test_densenet_121(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-12.onnx",
        artifacts_subdir="vision/classification",
    )


@pytest.mark.xfail(raises=IreeCompileException)
def test_efficientnet_lite4(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        artifacts_subdir="vision/classification",
    )


def test_googlenet(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx",
        artifacts_subdir="vision/classification",
    )


@pytest.mark.xfail(raises=IreeCompileException)
def test_inception_v1(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-12.onnx",
        artifacts_subdir="vision/classification",
    )


def test_inception_v2(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx",
        artifacts_subdir="vision/classification",
    )


def test_mnist(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx",
        artifacts_subdir="vision/classification",
    )


@pytest.mark.xfail(raises=IreeRunException)
def test_mobilenet(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
        artifacts_subdir="vision/classification",
    )


@pytest.mark.xfail(raises=IreeCompileException)
def test_rcnn_ilsvrc13(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.onnx",
        artifacts_subdir="vision/classification",
    )


def test_resnet50_v1(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx",
        artifacts_subdir="vision/classification",
    )


@pytest.mark.xfail(raises=IreeCompileException)
def test_resnet50_v2(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx",
        artifacts_subdir="vision/classification",
    )


def test_shufflenet(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-9.onnx",
        artifacts_subdir="vision/classification",
    )


def test_shufflenet_v2(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-v2-12.onnx",
        artifacts_subdir="vision/classification",
    )


def test_squeezenet(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-9.onnx",
        artifacts_subdir="vision/classification",
    )


@pytest.mark.size_medium
def test_vgg19(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg19-7.onnx",
        artifacts_subdir="vision/classification",
    )


@pytest.mark.size_medium
@pytest.mark.xfail(raises=IreeCompileException)
def test_zfnet_512(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/zfnet-512/model/zfnet512-12.onnx",
        artifacts_subdir="vision/classification",
    )
