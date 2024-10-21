func.func @conv2d_accumulate_2_2_32_32_times_3_3_2_dtype_f16_f16_f16(%lhs: tensor<2x2x32x32xf16>, %rhs: tensor<2x2x3x3xf16>, %acc: tensor<2x2x30x30xf16>) -> tensor<2x2x30x30xf16> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x2x32x32xf16>, tensor<2x2x3x3xf16>) outs(%acc: tensor<2x2x30x30xf16>) -> tensor<2x2x30x30xf16>
  return %result: tensor<2x2x30x30xf16>
}

func.func @conv2d_accumulate_2_2_32_32_times_3_3_64_dtype_f16_f16_f16(%lhs: tensor<2x2x32x32xf16>, %rhs: tensor<64x2x3x3xf16>, %acc: tensor<2x64x30x30xf16>) -> tensor<2x64x30x30xf16> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x2x32x32xf16>, tensor<64x2x3x3xf16>) outs(%acc: tensor<2x64x30x30xf16>) -> tensor<2x64x30x30xf16>
  return %result: tensor<2x64x30x30xf16>
}

func.func @conv2d_accumulate_2_32_32_32_times_3_3_64_dtype_f16_f16_f16(%lhs: tensor<2x32x32x32xf16>, %rhs: tensor<64x32x3x3xf16>, %acc: tensor<2x64x30x30xf16>) -> tensor<2x64x30x30xf16> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x32x32x32xf16>, tensor<64x32x3x3xf16>) outs(%acc: tensor<2x64x30x30xf16>) -> tensor<2x64x30x30xf16>
  return %result: tensor<2x64x30x30xf16>
}

