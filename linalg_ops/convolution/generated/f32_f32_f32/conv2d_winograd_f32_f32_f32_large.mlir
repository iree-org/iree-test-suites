func.func @conv2d_accumulate_2_4_128_128_times_3_3_8_dtype_f32_f32_f32(%lhs: tensor<2x4x128x128xf32>, %rhs: tensor<8x4x3x3xf32>, %acc: tensor<2x8x126x126xf32>) -> tensor<2x8x126x126xf32> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x4x128x128xf32>, tensor<8x4x3x3xf32>) outs(%acc: tensor<2x8x126x126xf32>) -> tensor<2x8x126x126xf32>
  return %result: tensor<2x8x126x126xf32>
}

func.func @conv2d_accumulate_2_3_128_128_times_3_3_12_dtype_f32_f32_f32(%lhs: tensor<2x3x128x128xf32>, %rhs: tensor<12x3x3x3xf32>, %acc: tensor<2x12x126x126xf32>) -> tensor<2x12x126x126xf32> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x3x128x128xf32>, tensor<12x3x3x3xf32>) outs(%acc: tensor<2x12x126x126xf32>) -> tensor<2x12x126x126xf32>
  return %result: tensor<2x12x126x126xf32>
}

