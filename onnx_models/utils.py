# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class IreeImportOnnxException(RuntimeError):
    pass


class IreeCompileException(RuntimeError):
    pass


class IreeRunException(RuntimeError):
    pass
