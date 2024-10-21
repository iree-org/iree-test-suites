#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script runs generate_e2e_conv2d_tests for all argument combinations that
# we are interested in testing.
#
# The output is a 'generated' folder with contents like this:
#   linalg_ops/
#     convolution/
#       generated/
#         f16_f16_f16/
#           conv2d_f16_f16_f16_large_calls.mlir
#           conv2d_f16_f16_f16_large.mlir
#           conv2d_f16_f16_f16_medium_calls.mlir
#           conv2d_f16_f16_f16_medium.mlir
#           conv2d_f16_f16_f16_small_calls.mlir
#           conv2d_f16_f16_f16_small.mlir
#           conv2d_winograd_f16_f16_f16_large_calls.mlir
#           conv2d_winograd_f16_f16_f16_large.mlir
#           conv2d_winograd_f16_f16_f16_medium_calls.mlir
#           conv2d_winograd_f16_f16_f16_medium.mlir
#           conv2d_winograd_f16_f16_f16_small_calls.mlir
#           conv2d_winograd_f16_f16_f16_small.mlir
#         f32_f32_f32/
#           conv2d_f32_f32_f32_large_calls.mlir
#           conv2d_f32_f32_f32_large.mlir
#           conv2d_f32_f32_f32_medium_calls.mlir
#           conv2d_f32_f32_f32_medium.mlir
#           conv2d_f32_f32_f32_small_calls.mlir
#           conv2d_f32_f32_f32_small.mlir
#           conv2d_winograd_f32_f32_f32_large_calls.mlir
#           conv2d_winograd_f32_f32_f32_large.mlir
#           conv2d_winograd_f32_f32_f32_medium_calls.mlir
#           conv2d_winograd_f32_f32_f32_medium.mlir
#           conv2d_winograd_f32_f32_f32_small_calls.mlir
#           conv2d_winograd_f32_f32_f32_small.mlir
#         ...
#           ...
# Usage:
#   generate_test_mlir_files.sh

set -euo pipefail

this_dir="$(cd $(dirname $0) && pwd)"
generated_dir_root="${this_dir}/generated"

# Reset generated directory.
rm -rf ${generated_dir_root?}
mkdir -p ${generated_dir_root?}

shapes=(
  "small"
  "medium"
  "large"
)

# input_type;kernel_type;acc_type
type_combinations=(
  "f16;f16;f16"
  "f32;f32;f32"
)

for type_combination in ${type_combinations[@]}; do
  IFS=";" read -r -a types <<< "${type_combination}"
  input_type="${types[0]}"
  kernel_type="${types[1]}"
  acc_type="${types[2]}"

  type_name="${input_type}_${kernel_type}_${acc_type}"
  type_combination_dir="${generated_dir_root}/${type_name}"
  mkdir -p ${type_combination_dir}

  for shape in ${shapes[@]}; do
    echo "Generating conv2d test files for ${type_name}_${shape}"

    name="conv2d_${type_name}_${shape}"
    python ${this_dir}/generate_e2e_conv2d_tests.py \
      --output_conv2d_mlir=${type_combination_dir}/${name}.mlir \
      --output_calls_mlir=${type_combination_dir}/${name}_calls.mlir \
      --input_type=${input_type} \
      --kernel_type=${kernel_type} \
      --acc_type=${acc_type} \
      --shapes=${shape}
  done
done
