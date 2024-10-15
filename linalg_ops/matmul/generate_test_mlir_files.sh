#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script runs generate_e2e_matmul_tests for all argument combinations that
# we are interested in testing.
#
# The output is a 'generated' folder with contents like this:
#   linalg_ops/
#     matmul/
#       generated/
#         f16_into_f16/
#           matmul_f16_into_f16_large_calls.mlir
#           matmul_f16_into_f16_large.mlir
#           matmul_f16_into_f16_small_calls.mlir
#           matmul_f16_into_f16_small.mlir
#           ...
#           matmul_transpose_b_f16_into_f16_large_calls.mlir
#           matmul_transpose_b_f16_into_f16_large.mlir
#           matmul_transpose_b_f16_into_f16_small_calls.mlir
#           matmul_transpose_b_f16_into_f16_small.mlir
#         f16_into_f32/
#           ...
#         f32_into_f32
#           ...
#         ...
#
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
  "large"
)

# lhs_rhs_type;accumulator_type
type_combinations=(
  "i8;i32"
  "f32;f32"
  "f16;f16"
  "f16;f32"
  "bf16;bf16"
  "bf16;f32"
  "f8E4M3FNUZ;f32"
)

for type_combination in ${type_combinations[@]}; do
  IFS=";" read -r -a types <<< "${type_combination}"
  lhs_rhs_type="${types[0]}"
  acc_type="${types[1]}"
  type_name="${lhs_rhs_type}_into_${acc_type}"

  type_combination_dir="${generated_dir_root}/${type_name}"
  mkdir -p ${type_combination_dir}

  for shape in ${shapes[@]}; do
    echo "Generating matmul test files for ${type_name}_${shape}"

    name="matmul_${type_name}_${shape}"
    python ${this_dir}/generate_e2e_matmul_tests.py \
      --output_matmul_mlir=${type_combination_dir}/${name}.mlir \
      --output_calls_mlir=${type_combination_dir}/${name}_calls.mlir \
      --lhs_rhs_type=${lhs_rhs_type} \
      --acc_type=${acc_type} \
      --shapes=${shape}

    name="matmul_transpose_b_${type_name}_${shape}"
    python ${this_dir}/generate_e2e_matmul_tests.py \
      --output_matmul_mlir=${type_combination_dir}/${name}.mlir \
      --output_calls_mlir=${type_combination_dir}/${name}_calls.mlir \
      --lhs_rhs_type=${lhs_rhs_type} \
      --acc_type=${acc_type} \
      --shapes=${shape} \
      --transpose_rhs
  done
done
