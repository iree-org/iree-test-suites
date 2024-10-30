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
#         f16_nchw_f16_fchw_f16/
#           conv2d_f16_nchw_f16_fchw_f16_large_calls.mlir
#           conv2d_f16_nchw_f16_fchw_f16_large.mlir
#           conv2d_f16_nchw_f16_fchw_f16_medium_calls.mlir
#           conv2d_f16_nchw_f16_fchw_f16_medium.mlir
#           conv2d_f16_nchw_f16_fchw_f16_small_calls.mlir
#           conv2d_f16_nchw_f16_fchw_f16_small.mlir
#         f16_nchw_f16_fchw_f32/
#           conv2d_f16_nchw_f16_fchw_f32_large_calls.mlir
#           conv2d_f16_nchw_f16_fchw_f32_large.mlir
#           conv2d_f16_nchw_f16_fchw_f32_medium_calls.mlir
#           conv2d_f16_nchw_f16_fchw_f32_medium.mlir
#           conv2d_f16_nchw_f16_fchw_f32_small_calls.mlir
#           conv2d_f16_nchw_f16_fchw_f32_small.mlir
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

# input_type;input_layout;kernel_type;kernel_layout;acc_type
type_and_layout_combinations=(
  "f16;nhwc;f16;hwcf;f16"
  "f16;nchw;f16;fchw;f16"
  "f16;nhwc;f16;hwcf;f32"
  "f16;nchw;f16;fchw;f32"
  "f32;nhwc;f32;hwcf;f32"
  "f32;nchw;f32;fchw;f32"
  "i8;nhwc;i8;hwcf;i32"
  "i8;nchw;i8;fchw;i32"
)

for type_and_layout_combination in ${type_and_layout_combinations[@]}; do
  IFS=";" read -r -a combination <<< "${type_and_layout_combination}"
  input_type="${combination[0]}"
  input_layout="${combination[1]}"
  kernel_type="${combination[2]}"
  kernel_layout="${combination[3]}"
  acc_type="${combination[4]}"

  type_layout_name="${input_type}_${input_layout}_${kernel_type}_${kernel_layout}_${acc_type}"
  type_combination_dir="${generated_dir_root}/${type_layout_name}"
  mkdir -p ${type_combination_dir}
  
  for shape in ${shapes[@]}; do
    echo "Generating conv2d test files for ${type_layout_name}_${shape}"
    name="conv2d_${type_layout_name}_${shape}"
    python ${this_dir}/generate_e2e_conv2d_tests.py \
      --output_conv2d_mlir=${type_combination_dir}/${name}.mlir \
      --output_calls_mlir=${type_combination_dir}/${name}_calls.mlir \
      --input_type=${input_type} \
      --input_layout=${input_layout} \
      --kernel_type=${kernel_type} \
      --kernel_layout=${kernel_layout} \
      --acc_type=${acc_type} \
      --shapes=${shape}
  done
done
