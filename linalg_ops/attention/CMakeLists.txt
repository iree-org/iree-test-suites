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


set(_DTYPES_AND_LAYOUTS)
list(APPEND _DTYPES_AND_LAYOUTS "f16_f16_f16_f16")

###############################################################################
#
# CPU - llvm-cpu on local-task, default flags.
#
###############################################################################

foreach(_DTYPE_AND_LAYOUT IN LISTS _DTYPES_AND_LAYOUTS)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        attention_llvm-cpu_local-task_${_DTYPE_AND_LAYOUT}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/attention_${_DTYPE_AND_LAYOUT}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/attention_${_DTYPE_AND_LAYOUT}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-attention-test
      TARGET_BACKEND
        "llvm-cpu"
      DRIVER
        "local-task"
      COMPILER_FLAGS
        "--iree-llvmcpu-target-cpu=host"
      RUNNER_FLAGS
      LABELS
         "hostonly"
         "local"
    )
  endforeach()
endforeach()

###############################################################################
#
# GPU - ROCm/HIP, CDNA(gfx9).
#
###############################################################################

# To distinguish between CDNA(gfx9) and RDNA3(gfx11)
if(IREE_HIP_TEST_TARGET_CHIP MATCHES "^gfx9")

foreach(_DTYPE_AND_LAYOUT IN LISTS _DTYPES_AND_LAYOUTS)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        attention_rocm_hip_${_DTYPE_AND_LAYOUT}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/attention_${_DTYPE_AND_LAYOUT}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/attention_${_DTYPE_AND_LAYOUT}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-attention-test
      TARGET_BACKEND
        "rocm"
      DRIVER
        "hip"
      COMPILER_FLAGS
        "--iree-hip-target=${IREE_HIP_TEST_TARGET_CHIP}"
      RUNNER_FLAGS
      LABELS
    )
  endforeach()
endforeach()

###############################################################################
#
# GPU - ROCm/HIP, CDNA(gfx11)
#
###############################################################################

elseif(IREE_HIP_TEST_TARGET_CHIP MATCHES "^gfx11")

foreach(_DTYPE_AND_LAYOUT IN LISTS _DTYPES_AND_LAYOUTS)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        attention_rocm_hip_${_DTYPE_AND_LAYOUT}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/attention_${_DTYPE_AND_LAYOUT}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/attention_${_DTYPE_AND_LAYOUT}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-attention-test
      TARGET_BACKEND
        "rocm"
      DRIVER
        "hip"
      COMPILER_FLAGS
        "--iree-hip-target=${IREE_HIP_TEST_TARGET_CHIP}"
      RUNNER_FLAGS
      LABELS
    )
  endforeach()
endforeach()

endif()
