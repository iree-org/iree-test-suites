# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

#-------------------------------------------------------------------------------
# Project configuration
#-------------------------------------------------------------------------------

project(iree-test-suites-linalg C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Baseline requirements.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

#-------------------------------------------------------------------------------
# Core project dependency
#-------------------------------------------------------------------------------

option(IREE_USE_LOCAL_REPO "Uses a local repository instead of fetching on-demand." OFF)
set(IREE_LOCAL_REPO_PATH "" CACHE STRING "Local repository path")

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree-test-suites")
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_SAMPLES OFF)
set(IREE_BUILD_TESTS OFF)

if(IREE_USE_LOCAL_REPO)
  message(STATUS "Using IREE repo at path '${IREE_LOCAL_REPO_PATH}'")
  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  add_subdirectory(${IREE_LOCAL_REPO_PATH} ${CMAKE_CURRENT_BINARY_DIR}/iree EXCLUDE_FROM_ALL)
  list(POP_BACK CMAKE_MESSAGE_INDENT)
else()
  message(STATUS "Fetching the core IREE repo (this may take a few minutes)...")
  list(APPEND CMAKE_MESSAGE_INDENT "  ")

  # Note: for log output, set -DFETCHCONTENT_QUIET=OFF,
  # see https://gitlab.kitware.com/cmake/cmake/-/issues/18238#note_440475
  include(FetchContent)

  FetchContent_Declare(
    iree
    GIT_REPOSITORY https://github.com/iree-org/iree.git
    GIT_TAG candidate-20240828.999
    GIT_SUBMODULES_RECURSE OFF
    GIT_SHALLOW OFF
    GIT_PROGRESS ON
    USES_TERMINAL_DOWNLOAD ON
    EXCLUDE_FROM_ALL
  )

  FetchContent_MakeAvailable(iree)
  FetchContent_GetProperties(iree SOURCE_DIR IREE_SOURCE_DIR)

  list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

#-------------------------------------------------------------------------------
# Test code
#-------------------------------------------------------------------------------

enable_testing(iree-test-suites-linalg-ops)

add_custom_target(iree-test-suites-linalg-ops-deps
  COMMENT
    "Building linalg operator test suite deps"
)

iree_cc_library(
  NAME
    test_utils
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
    ::test_utils
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
    ::test_utils
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

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
include(iree_test_suites_native_test)
include(iree_test_suites_runner_test)

add_subdirectory(matmul)
add_subdirectory(convolution)