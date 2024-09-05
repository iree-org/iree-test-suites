# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_test_suites_runner_test(
  NAME
    matmul_cpu_f32_f32_small
  TESTS_SRC
    "generated/f16_into_f32/matmul_f16_into_f32_small.mlir"
  CALLS_SRC
    "generated/f16_into_f32/matmul_f16_into_f32_small_calls.mlir"
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
