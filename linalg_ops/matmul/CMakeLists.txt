# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO(scotttodd): add filtering here, in the helper functions, or in ctest to
#                  choose which tests to compile and run

set(_SIZES)
list(APPEND _SIZES "large")
list(APPEND _SIZES "small")

###############################################################################
#
# CPU - llvm-cpu on local-task, default flags.
#
###############################################################################

set(_DTYPES)
list(APPEND _DTYPES "i8_into_i32")
list(APPEND _DTYPES "f32_into_f32")
list(APPEND _DTYPES "f16_into_f16")
list(APPEND _DTYPES "f16_into_f32")
list(APPEND _DTYPES "bf16_into_bf16")
list(APPEND _DTYPES "bf16_into_f32")
# list(APPEND _DTYPES "f8E4M3FNUZ_into_f32")  # Unsupported data type.
foreach(_DTYPE IN LISTS _DTYPES)
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

###############################################################################
#
# CPU - vmvx on local-task, default flags.
#
###############################################################################

set(_DTYPES)
list(APPEND _DTYPES "i8_into_i32")
list(APPEND _DTYPES "f32_into_f32")
# list(APPEND _DTYPES "f16_into_f16")  # Unsupported data type.
# list(APPEND _DTYPES "f16_into_f32")  # Unsupported data type.
# list(APPEND _DTYPES "bf16_into_bf16")  # Unsupported data type.
# list(APPEND _DTYPES "bf16_into_f32")  # Unsupported data type.
# list(APPEND _DTYPES "f8E4M3FNUZ_into_f32")  # Unsupported data type.
foreach(_DTYPE IN LISTS _DTYPES)
  # Note: not running large tests on vmvx, too slow.
  set(_SIZE "small")
  iree_test_suites_runner_test(
    NAME
      matmul_vmvx_${_DTYPE}_${_SIZE}
    TESTS_SRC
      "generated/${_DTYPE}/matmul_${_DTYPE}_${_SIZE}.mlir"
    CALLS_SRC
      "generated/${_DTYPE}/matmul_${_DTYPE}_${_SIZE}_calls.mlir"
    TEST_RUNNER
      iree-test-suites_iree-e2e-matmul-test
    TARGET_BACKEND
      "vmvx"
    DRIVER
      "local-task"
    COMPILER_ARGS
    RUNNER_ARGS
    LABELS
  )
endforeach()

###############################################################################
#
# GPU - Vulkan, default flags.
#
###############################################################################

set(_DTYPES)
# list(APPEND _DTYPES "i8_into_i32")  # Currently failing.
list(APPEND _DTYPES "f32_into_f32")
# list(APPEND _DTYPES "f16_into_f16")  # Failing to compile.
# list(APPEND _DTYPES "f16_into_f32")  # Failing to compile.
# list(APPEND _DTYPES "bf16_into_bf16")  # Failing to compile.
# list(APPEND _DTYPES "bf16_into_f32")  # Failing to compile.
# list(APPEND _DTYPES "f8E4M3FNUZ_into_f32")  # Unsupported data type.
foreach(_DTYPE IN LISTS _DTYPES)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        matmul_vulkan_${_DTYPE}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE}/matmul_${_DTYPE}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE}/matmul_${_DTYPE}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-matmul-test
      TARGET_BACKEND
        "vulkan-spirv"
      DRIVER
        "vulkan"
      COMPILER_ARGS
      RUNNER_ARGS
      LABELS
    )
  endforeach()
endforeach()

###############################################################################
#
# GPU - CUDA, default flags.
#
###############################################################################

set(_DTYPES)
list(APPEND _DTYPES "i8_into_i32")
list(APPEND _DTYPES "f32_into_f32")
# list(APPEND _DTYPES "f16_into_f16")  # Timeout running.
list(APPEND _DTYPES "f16_into_f32")
# list(APPEND _DTYPES "bf16_into_bf16")  # Timeout running.
list(APPEND _DTYPES "bf16_into_f32")
# list(APPEND _DTYPES "f8E4M3FNUZ_into_f32")  # Unsupported data type.
foreach(_DTYPE IN LISTS _DTYPES)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        matmul_cuda_${_DTYPE}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE}/matmul_${_DTYPE}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE}/matmul_${_DTYPE}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-matmul-test
      TARGET_BACKEND
        "cuda"
      DRIVER
        "cuda"
      COMPILER_ARGS
      RUNNER_ARGS
      LABELS
    )
  endforeach()
endforeach()

###############################################################################
#
# GPU - ROCm/HIP, default flags.
#
###############################################################################

set(_DTYPES)
list(APPEND _DTYPES "i8_into_i32")
list(APPEND _DTYPES "f32_into_f32")
list(APPEND _DTYPES "f16_into_f16")
list(APPEND _DTYPES "f16_into_f32")
list(APPEND _DTYPES "bf16_into_bf16")
list(APPEND _DTYPES "bf16_into_f32")
# list(APPEND _DTYPES "f8E4M3FNUZ_into_f32")  # Failing to compile.
foreach(_DTYPE IN LISTS _DTYPES)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        matmul_hip_${_DTYPE}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE}/matmul_${_DTYPE}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE}/matmul_${_DTYPE}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-matmul-test
      TARGET_BACKEND
        "rocm"
      DRIVER
        "hip"
      COMPILER_ARGS
        "--iree-hip-target=${IREE_HIP_TEST_TARGET_CHIP}"
      RUNNER_ARGS
      LABELS
    )
  endforeach()
endforeach()
