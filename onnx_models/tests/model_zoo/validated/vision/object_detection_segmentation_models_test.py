# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/

import pytest

from .....utils import *

artifacts_subdir = "model_zoo/validated/vision/object_detection_segmentation"


@pytest.mark.size_large
def test_duc(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/duc/model/ResNet101-DUC-12.onnx",
        artifacts_subdir=artifacts_subdir,
    )


def test_faster_rcnn(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx",
        artifacts_subdir=artifacts_subdir,
    )


def test_fcn(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12.onnx",
        artifacts_subdir=artifacts_subdir,
    )


def test_mask_rcnn(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx",
        artifacts_subdir=artifacts_subdir,
    )


def test_retinanet(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx",
        artifacts_subdir=artifacts_subdir,
    )


def test_ssd(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd/model/ssd-12.onnx",
        artifacts_subdir=artifacts_subdir,
    )


@pytest.mark.xfail(raises=NotImplementedError)  # numpy.uint8
def test_ssd_mobilenetv1(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx",
        artifacts_subdir=artifacts_subdir,
    )


def test_tiny_yolov2(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx",
        artifacts_subdir=artifacts_subdir,
    )


@pytest.mark.skip("ONNXRuntimeError")
def test_tiny_yolov3(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx",
        artifacts_subdir=artifacts_subdir,
    )


def test_yolov2_coco(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx",
        artifacts_subdir=artifacts_subdir,
    )


@pytest.mark.skip("ONNXRuntimeError")
def test_yolov3(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolov3/model/yolov3-12.onnx",
        artifacts_subdir=artifacts_subdir,
    )


def test_yolov4(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolov4/model/yolov4.onnx",
        artifacts_subdir=artifacts_subdir,
    )
