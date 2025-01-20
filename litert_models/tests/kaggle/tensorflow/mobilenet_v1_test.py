# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://www.kaggle.com/models/tensorflow/mobilenet-v1/

import pytest

from ....utils import *


# https://www.kaggle.com/models/tensorflow/mobilenet-v1/tfLite/0-25-224
@pytest.mark.xfail(raises=IreeCompileException)
def test_mobilenet_v1_0_25_224(tflite_import_and_iree_compile):
    tflite_import_and_iree_compile("tensorflow/mobilenet-v1/tfLite/0-25-224")


# https://www.kaggle.com/models/tensorflow/mobilenet-v1/tfLite/0-25-224-quantized/
@pytest.mark.xfail(raises=IreeCompileException)
def test_mobilenet_v1_0_25_224_quantized(tflite_import_and_iree_compile):
    tflite_import_and_iree_compile("tensorflow/mobilenet-v1/tfLite/0-25-224-quantized")
