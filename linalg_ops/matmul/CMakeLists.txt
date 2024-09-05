# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_SIZES)
list(APPEND _SIZES "large")
list(APPEND _SIZES "small")

###############################################################################
# CPU (llvm-cpu on local-task), default flags.
set(_CPU_TEST_DTYPES)
list(APPEND _CPU_TEST_DTYPES "i8_into_i32")
list(APPEND _CPU_TEST_DTYPES "f32_into_f32")
list(APPEND _CPU_TEST_DTYPES "f16_into_f16")
list(APPEND _CPU_TEST_DTYPES "f16_into_f32")
list(APPEND _CPU_TEST_DTYPES "bf16_into_bf16")
list(APPEND _CPU_TEST_DTYPES "bf16_into_f32")
# list(APPEND _CPU_TEST_DTYPES "f8E4M3FNUZ_into_f32")
foreach(_DTYPE IN LISTS _CPU_TEST_DTYPES)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        matmul_cpu_${_DTYPE}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE}/matmul_${_DTYPE}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE}/matmul_${_DTYPE}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-matmul-test
      TARGET_BACKEND
        "llvm-cpu"
      DRIVER
        "local-task"
      COMPILER_ARGS
      RUNNER_ARGS
      LABELS
    )
  endforeach()
endforeach()
