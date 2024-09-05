# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_test_suites_native_test()
#
# Creates a test that runs the specified binary with the specified arguments.
#
# Parameters:
# NAME: name of target
# DRIVER: If specified, will pass --device=DRIVER to the test binary and adds
#     a driver label to the test.
#     TODO(scotttodd): Remove automatic args/labels, push those up a level
# DATA: Additional input files needed by the test binary.
# ARGS: additional arguments passed to the test binary.
#     --device=DRIVER is automatically added if specified.
#     File-related arguments can be passed with `{{}}` locator,
#     e.g., --input=@{{foo.npy}}. The locator is used to portably
#     pass the file arguments to tests and add the file to DATA.
# SRC: binary target to run as the test.
# WILL_FAIL: The target will run, but its pass/fail status will be inverted.
# DISABLED: The target will be skipped and its status will be 'Not Run'.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
# TIMEOUT: Test target timeout in seconds.
#
# Note: the DATA argument is not actually adding dependencies because CMake
# doesn't have a good way to specify a data dependency for a test.
#
# Usage:
# iree_cc_binary(
#   NAME
#     requires_args_to_run
#   ...
# )
# iree_test_suites_native_test(
#   NAME
#     requires_args_to_run_test
#   ARGS
#    --do-the-right-thing
#   SRC
#     ::requires_args_to_run
# )

function(iree_test_suites_native_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC;DRIVER;WILL_FAIL;DISABLED"
    "ARGS;LABELS;DATA;TIMEOUT"
    ${ARGN}
  )

  # Prefix the test with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  iree_package_ns(_PACKAGE_NS)
  iree_package_path(_PACKAGE_PATH)
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")

  # If driver was specified, add the corresponding test arg and label.
  if(DEFINED _RULE_DRIVER)
    list(APPEND _RULE_ARGS "--device=${_RULE_DRIVER}")
    list(APPEND _RULE_LABELS "driver=${_RULE_DRIVER}")
  endif()

  # Detect file location with `{{}}` and handle its portability for all entries
  # in `_RULE_ARGS`.
  foreach(_ARG ${_RULE_ARGS})
    string(REGEX MATCH ".*{{(.+)}}" _FILE_ARG "${_ARG}")
    if(_FILE_ARG)
      set(_FILE_PATH ${CMAKE_MATCH_1})
      list(APPEND _RULE_DATA "${_FILE_PATH}")
      # remove the `{{}}` from `_ARG` and append it to `_TEST_ARGS`.
      string(REGEX REPLACE "{{.+}}" "" _FILE_FLAG_PREFIX "${_ARG}")
      list(APPEND _TEST_ARGS "${_FILE_FLAG_PREFIX}${_FILE_PATH}")
    else()  # naive append
      list(APPEND _TEST_ARGS "${_ARG}")
    endif(_FILE_ARG)
  endforeach(_ARG)

  # Replace binary passed by relative ::name with iree::package::name
  string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _SRC_TARGET ${_RULE_SRC})

  add_test(
    NAME
      ${_TEST_NAME}
    COMMAND
      "$<TARGET_FILE:${_SRC_TARGET}>"
      ${_TEST_ARGS}
  )

  # File extension cmake uses for the target platform.
  set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "IREE_DYLIB_EXT=${CMAKE_SHARED_LIBRARY_SUFFIX}")

  if (NOT DEFINED _RULE_TIMEOUT)
    set(_RULE_TIMEOUT 60)
  endif()

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_TEST_NAME} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST "${_TEST_NAME}" PROPERTY REQUIRED_FILES "${_RULE_DATA}")
  set_property(TEST ${_TEST_NAME} PROPERTY TIMEOUT ${_RULE_TIMEOUT})
  if(_RULE_WILL_FAIL)
    set_property(TEST ${_TEST_NAME} PROPERTY WILL_FAIL ${_RULE_WILL_FAIL})
  endif()
  if(_RULE_DISABLED)
    set_property(TEST ${_TEST_NAME} PROPERTY DISABLED ${_RULE_DISABLED})
  endif()
endfunction()
