# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet
def test_mobilenetv2_12(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_name="mobilenetv2-12",
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
        input_name="input",
        input_shape=(1, 3, 224, 224),
        input_type="",
        output_name="output",
        output_shape=(),
        output_type="",
    )


# https://github.com/onnx/models/tree/main/validated/vision/classification/resnet
def test_resnet50_v1_12(compare_between_iree_and_onnxruntime):
    compare_between_iree_and_onnxruntime(
        model_name="resnet50-v1-12",
        model_url="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx",
        input_name="data",
        input_shape=(1, 3, 224, 224),
        input_type="",
        output_name="resnetv17_dense0_fwd",
        output_shape=(),
        output_type="",
    )


# TODO(scotttodd): add annotations:
#    xfail (with Exception subclass / reason)
#    marks (size of test, hardware required, etc.)
