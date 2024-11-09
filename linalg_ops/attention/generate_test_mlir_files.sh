#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script runs generate_e2e_attention_tests for all argument combinations that
# we are interested in testing.
#
# The output is a 'generated' folder with contents like this:
#   linalg_ops/
#     convolution/
#       generated/
#         f16_f16_f16_f16/
#           attention_f16_f16_f16_f16_large_calls.mlir
#           attention_f16_f16_f16_f16_large.mlir
#           attention_f16_f16_f16_f16_medium_calls.mlir
#           attention_f16_f16_f16_f16_medium.mlir
#           attention_f16_f16_f16_f16_small_calls.mlir
#           attention_f16_f16_f16_f16_small.mlir
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

# query_type;key_type;value_type;scale_type
type_and_layout_combinations=(
  "f16;f16;f16;f16"
)

for type_and_layout_combination in ${type_and_layout_combinations[@]}; do
  IFS=";" read -r -a combination <<< "${type_and_layout_combination}"
  query_type="${combination[0]}"
  key_type="${combination[1]}"
  value_type="${combination[2]}"
  scale_type="${combination[3]}"

  type_layout_name="${query_type}_${key_type}_${value_type}_${scale_type}"
  type_combination_dir="${generated_dir_root}/${type_layout_name}"
  mkdir -p ${type_combination_dir}
  
  for shape in ${shapes[@]}; do
    echo "Generating attention test files for ${type_layout_name}_${shape}"
    name="attention_${type_layout_name}_${shape}"
    python ${this_dir}/generate_e2e_attention_tests.py \
      --output_attention_mlir=${type_combination_dir}/${name}.mlir \
      --output_calls_mlir=${type_combination_dir}/${name}_calls.mlir \
      --query_type=${query_type} \
      --key_type=${key_type} \
      --value_type=${value_type} \
      --shapes=${shape}
  done
done
