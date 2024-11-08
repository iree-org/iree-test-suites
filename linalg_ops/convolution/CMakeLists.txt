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
list(APPEND _DTYPES_AND_LAYOUTS "f16_nhwc_f16_hwcf_f16")
list(APPEND _DTYPES_AND_LAYOUTS "f16_nchw_f16_fchw_f16")
list(APPEND _DTYPES_AND_LAYOUTS "f16_nhwc_f16_hwcf_f32")
list(APPEND _DTYPES_AND_LAYOUTS "f16_nchw_f16_fchw_f32")
list(APPEND _DTYPES_AND_LAYOUTS "f32_nhwc_f32_hwcf_f32")
list(APPEND _DTYPES_AND_LAYOUTS "f32_nchw_f32_fchw_f32")

###############################################################################
#
# CPU - llvm-cpu on local-task, default flags.
#
###############################################################################

foreach(_DTYPE_AND_LAYOUT IN LISTS _DTYPES_AND_LAYOUTS)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        conv2d_llvm-cpu_local-task_${_DTYPE_AND_LAYOUT}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-conv2d-test
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
# CPU - Winograd llvm-cpu on local-task, default flags.
#
###############################################################################

foreach(_DTYPE_AND_LAYOUT IN LISTS _DTYPES_AND_LAYOUTS)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        conv2d_winograd_llvm-cpu_local-task_${_DTYPE_AND_LAYOUT}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}_calls.mlir"
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
# GPU - ROCm/HIP, CDNA(gfx9).
#
###############################################################################

# To distinguish between CDNA(gfx9) and RDNA3(gfx11)
if(IREE_HIP_TEST_TARGET_CHIP MATCHES "^gfx9")

foreach(_DTYPE_AND_LAYOUT IN LISTS _DTYPES_AND_LAYOUTS)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        conv2d_rocm_hip_${_DTYPE_AND_LAYOUT}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-conv2d-test
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
# Winograd GPU - ROCm/HIP, CDNA(gfx9).
#
###############################################################################

foreach(_DTYPE_AND_LAYOUT IN LISTS _DTYPES_AND_LAYOUTS)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        conv2d_winograd_rocm_hip_${_DTYPE_AND_LAYOUT}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-conv2d-test
      TARGET_BACKEND
        "rocm"
      DRIVER
        "hip"
      COMPILER_FLAGS
        "--iree-preprocessing-pass-pipeline=builtin.module\(func.func\(iree-linalg-ext-convert-conv2d-to-winograd{replace-all-convs=true}\)\)"
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
        conv2d_rocm_hip_${_DTYPE_AND_LAYOUT}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-conv2d-test
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
# Winograd GPU - ROCm/HIP, CDNA(gfx11).
#
###############################################################################

foreach(_DTYPE_AND_LAYOUT IN LISTS _DTYPES_AND_LAYOUTS)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        conv2d_winograd_rocm_hip_${_DTYPE_AND_LAYOUT}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-conv2d-test
      TARGET_BACKEND
        "rocm"
      DRIVER
        "hip"
      COMPILER_FLAGS
        "--iree-preprocessing-pass-pipeline=builtin.module\(func.func\(iree-linalg-ext-convert-conv2d-to-winograd{replace-all-convs=true}\)\)"
        "--iree-hip-target=${IREE_HIP_TEST_TARGET_CHIP}"
      RUNNER_FLAGS
      LABELS
    )
  endforeach()
endforeach()

endif()

# CPU and GPU tests for without Winograd
set(_DTYPES_AND_LAYOUTS)
list(APPEND _DTYPES_AND_LAYOUTS "i8_nhwc_i8_hwcf_i32")
list(APPEND _DTYPES_AND_LAYOUTS "i8_nchw_i8_fchw_i32")

###############################################################################
#
# CPU - llvm-cpu on local-task, default flags.
#
###############################################################################

foreach(_DTYPE_AND_LAYOUT IN LISTS _DTYPES_AND_LAYOUTS)
  foreach(_SIZE IN LISTS _SIZES)
    iree_test_suites_runner_test(
      NAME
        conv2d_llvm-cpu_local-task_${_DTYPE_AND_LAYOUT}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-conv2d-test
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
        conv2d_rocm_hip_${_DTYPE_AND_LAYOUT}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-conv2d-test
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
        conv2d_rocm_hip_${_DTYPE_AND_LAYOUT}_${_SIZE}
      TESTS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}.mlir"
      CALLS_SRC
        "generated/${_DTYPE_AND_LAYOUT}/conv2d_${_DTYPE_AND_LAYOUT}_${_SIZE}_calls.mlir"
      TEST_RUNNER
        iree-test-suites_iree-e2e-conv2d-test
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
