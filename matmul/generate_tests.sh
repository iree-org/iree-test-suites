#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

shapes=(
  "small"
  "large"
  "gpu_large"
  "gpu_large_aligned"
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

for shape in ${shapes[@]}; do
  for type_combination in ${type_combinations[@]}; do
    IFS=";" read -r -a types <<< "${type_combination}"
    lhs_rhs_type="${types[0]}"
    acc_type="${types[1]}"
    echo "Generating ${lhs_rhs_type}_${acc_type}_${shape}"

    name="matmul_${lhs_rhs_type}_${acc_type}_${shape}"
    python tests/generate_e2e_matmul_tests.py \
      --output_matmul_mlir=tests/generated/${name}_matmul.mlir \
      --output_calls_mlir=tests/generated/${name}_calls.mlir \
      --lhs_rhs_type=${lhs_rhs_type} \
      --acc_type=${acc_type} \
      --shapes=${shape}
  done
done
