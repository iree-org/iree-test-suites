# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_test_suites_runner_test()
#
# Creates a test using a specified test runner program for the specified
# test files.
#
# Parameters:
#   NAME: Name of the target
#   TESTS_SRC: MLIR source file to be compiled to an IREE module.
#   CALLS_SRC: MLIR source file with calls to be compiled to an IREE module.
#   TEST_RUNNER: Test runner program.
#   TARGET_BACKEND: Target backend to compile for.
#   DRIVER: Driver to run the module with.
#   COMPILER_FLAGS: additional args to pass to the compiler.
#       Target backend flags are passed automatically.
#   RUNNER_FLAGS: Additional args to pass to the runner program.
#       The device and input file flags are passed automatically.
#   LABELS: Additional labels to apply to the test.
#       "driver=${DRIVER}" is added automatically.
function(iree_test_suites_runner_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TESTS_SRC;CALLS_SRC;TEST_RUNNER;TARGET_BACKEND;DRIVER"
    "COMPILER_FLAGS;RUNNER_FLAGS;LABELS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  set(_BASE_COMPILER_FLAGS
    "--iree-hal-target-backends=${_RULE_TARGET_BACKEND}"
  )

  set(_TESTS_VMFB "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.vmfb")
  set(_CALLS_VMFB "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_calls.vmfb")

  iree_bytecode_module(
    NAME
      "${_RULE_NAME}_module"
    MODULE_FILE_NAME
      "${_TESTS_VMFB}"
    SRC
      "${_RULE_TESTS_SRC}"
    FLAGS
      "${_BASE_COMPILER_FLAGS}"
      "${_RULE_COMPILER_FLAGS}"
  )
  iree_bytecode_module(
    NAME
      "${_RULE_NAME}_calls_module"
    MODULE_FILE_NAME
      "${_CALLS_VMFB}"
    SRC
      "${_RULE_CALLS_SRC}"
    FLAGS
      "${_BASE_COMPILER_FLAGS}"
      "${_RULE_COMPILER_FLAGS}"
  )

  # A target specifically for the test. We could combine this with the above,
  # but we want that one to get pulled into iree_bytecode_module.
  add_custom_target("${_NAME}" ALL)
  add_dependencies(
    "${_NAME}"
    "${_NAME}_module"
    "${_NAME}_calls_module"
    "${_RULE_TEST_RUNNER}"
  )

  add_dependencies(iree-test-suites-linalg-ops-deps "${_NAME}")

  iree_test_suites_native_test(
    NAME
      "${_RULE_NAME}${_RULE_VARIANT_NAME}"
    DRIVER
      "${_RULE_DRIVER}"
    SRC
      "${_RULE_TEST_RUNNER}"
    DATA
      ${_TESTS_VMFB}
      ${_CALLS_VMFB}
    ARGS
      "--module={{${_TESTS_VMFB}}}"
      "--module={{${_CALLS_VMFB}}}"
      ${_RULE_RUNNER_FLAGS}
    LABELS
      ${_RULE_LABELS}
    DISABLED
      ${_RULE_TEST_DISABLED}
  )
endfunction()
