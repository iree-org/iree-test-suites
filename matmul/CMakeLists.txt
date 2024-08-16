# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

#-------------------------------------------------------------------------------
# Project configuration
#-------------------------------------------------------------------------------

project(iree-test-suites-matmul C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#-------------------------------------------------------------------------------
# Core project dependency
#-------------------------------------------------------------------------------

# TODO(scotttodd): make this configurable with an option
#   - point at local source dir (for testing in iree-org/iree or local dev)
#   - fallback to fetching

message(STATUS "Fetching core IREE repo (this may take a few minutes)...")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

# Note: for log output, set -DFETCHCONTENT_QUIET=OFF,
# see https://gitlab.kitware.com/cmake/cmake/-/issues/18238#note_440475
include(FetchContent)

# TODO(scotttodd): pin to a version from a stable release?
FetchContent_Declare(
  iree
  GIT_REPOSITORY https://github.com/iree-org/iree.git
  GIT_TAG f1319fc3404a2c97bfc76c6ee2268a96954a1b2c # 2024-08-16
  GIT_SUBMODULES_RECURSE OFF
  GIT_SHALLOW OFF
  GIT_PROGRESS ON
  EXCLUDE_FROM_ALL
  USES_TERMINAL_DOWNLOAD ON
)

# Disable core project feature not needed for these tests.
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)

FetchContent_MakeAvailable(iree)
FetchContent_GetProperties(iree SOURCE_DIR IREE_SOURCE_DIR)

list(POP_BACK CMAKE_MESSAGE_INDENT)

#-------------------------------------------------------------------------------
# Test code
#-------------------------------------------------------------------------------

if(NOT IREE_ENABLE_THREADING)
  return()
endif()

iree_cc_library(
  NAME
    e2e_test_util
  HDRS
    "test_utils.h"
  SRCS
    "test_utils.c"
  DEPS
    iree::base
    iree::base::internal
    iree::base::internal::cpu
    iree::base::internal::flags
    iree::base::internal::path
    iree::hal
    iree::modules::hal
    iree::tooling::context_util
    iree::tooling::device_util
    iree::vm
    iree::vm::cc
  PUBLIC
)

iree_cc_binary(
  NAME
    iree-e2e-matmul-test
  SRCS
    "iree-e2e-matmul-test.cc"
  DEPS
    ::e2e_test_util
    iree::base
    iree::base::internal
    iree::base::internal::cpu
    iree::base::internal::flags
    iree::base::internal::path
    iree::hal
    iree::modules::hal
    iree::tooling::context_util
    iree::tooling::device_util
    iree::vm
    iree::vm::cc
)

iree_cc_binary(
  NAME
    iree-e2e-conv2d-test
  SRCS
    "iree-e2e-conv2d-test.cc"
  DEPS
    ::e2e_test_util
    iree::base
    iree::base::internal
    iree::base::internal::cpu
    iree::base::internal::flags
    iree::base::internal::path
    iree::hal
    iree::modules::hal
    iree::tooling::context_util
    iree::tooling::device_util
    iree::vm
    iree::vm::cc
)

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

add_subdirectory(tests)
