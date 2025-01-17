# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/

import pytest

from .....utils import *

BASE_PATH = "validated/vision/object_detection_segmentation/"


@pytest.mark.parametrize(
    "model",
    [
        # fmt: off
        pytest.param("duc/model/ResNet101-DUC-12.onnx", marks=pytest.mark.size_large),
        pytest.param("faster-rcnn/model/FasterRCNN-12.onnx"),
        pytest.param("fcn/model/fcn-resnet50-12.onnx"),
        pytest.param("mask-rcnn/model/MaskRCNN-12.onnx"),
        pytest.param("retinanet/model/retinanet-9.onnx"),
        pytest.param("ssd/model/ssd-12.onnx"),
        pytest.param("ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx", marks=pytest.mark.xfail(raises=NotImplementedError)),
        pytest.param("tiny-yolov2/model/tinyyolov2-8.onnx"),
        pytest.param("tiny-yolov3/model/tiny-yolov3-11.onnx",marks=pytest.mark.skip("ONNXRuntimeError")),
        pytest.param("yolov2-coco/model/yolov2-coco-9.onnx"),
        pytest.param("yolov3/model/yolov3-12.onnx", marks=pytest.mark.skip("ONNXRuntimeError")),
        pytest.param("yolov4/model/yolov4.onnx"),
        # fmt: on
    ],
)
def test_models(compare_between_iree_and_onnxruntime, model):
    compare_between_iree_and_onnxruntime(
        model_url=BASE_PATH + model, artifacts_subdir=BASE_PATH
    )
