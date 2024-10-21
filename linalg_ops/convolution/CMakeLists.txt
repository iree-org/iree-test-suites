# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO(scotttodd): add filtering here, in the helper functions, or in ctest to
#                  choose which tests to compile and run

set(_SIZES)
list(APPEND _SIZES "large")
list(APPEND _SIZES "medium")
list(APPEND _SIZES "small")

###############################################################################
#
# CPU - llvm-cpu on local-task, default flags.
#
###############################################################################


set(_DTYPES)
list(APPEND _DTYPES "f16_f16_f16")
list(APPEND _DTYPES "f32_f32_f32")

foreach(_DTYPE IN LISTS _DTYPES)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        conv2d_${_DTYPE}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE}/conv2d_${_DTYPE}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE}/conv2d_${_DTYPE}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-conv2d-test
      TARGET_BACKEND
        "llvm-cpu"
      DRIVER
        "local-task"
      COMPILER_FLAGS
      RUNNER_FLAGS
      LABELS
         "hostonly"
         "local"
    )
  endforeach()
endforeach()

foreach(_DTYPE IN LISTS _DTYPES)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        conv2d_winograd_${_DTYPE}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE}/conv2d_${_DTYPE}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE}/conv2d_${_DTYPE}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-conv2d-test
      TARGET_BACKEND
        "llvm-cpu"
      DRIVER
        "local-task"
      COMPILER_FLAGS
        "--iree-preprocessing-pass-pipeline=builtin.module\(func.func\(iree-linalg-ext-convert-conv2d-to-winograd{replace-all-convs=true}\)\)"
      RUNNER_FLAGS
      LABELS
         "hostonly"
         "local"
      TARGET_CPU_FEATURES_VARIANTS
         "default"
    )
  endforeach()
endforeach()

###############################################################################
#
# GPU - ROCm/HIP, default flags.
#
###############################################################################

# if(IREE_HIP_TEST_TARGET_CHIP)

# set(_DTYPES)
# list(APPEND _DTYPES "f16_f16_f16")
# list(APPEND _DTYPES "f32_f32_f32")

# foreach(_DTYPE IN LISTS _DTYPES)
#   foreach(_SIZE IN LISTS _SIZES)
#     iree_test_suites_runner_test(
#       NAME
#         matmul_hip_${_DTYPE}_${_SIZE}
#       TESTS_SRC
#         "generated/${_DTYPE}/matmul_${_DTYPE}_${_SIZE}.mlir"
#       CALLS_SRC
#         "generated/${_DTYPE}/matmul_${_DTYPE}_${_SIZE}_calls.mlir"
#       TEST_RUNNER
#         iree-test-suites_iree-e2e-matmul-test
#       TARGET_BACKEND
#         "rocm"
#       DRIVER
#         "hip"
#       COMPILER_FLAGS
#         "--iree-hip-target=${IREE_HIP_TEST_TARGET_CHIP}"
#       RUNNER_FLAGS
#       LABELS
#     )
#   endforeach()
# endforeach()

# endif()