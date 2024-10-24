func.func @conv2d_accumulate_1_1_1_1_times_1_1_1_dtype_f16_f16_f32(%lhs: tensor<1x1x1x1xf16>, %rhs: tensor<1x1x1x1xf16>, %acc: tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>) outs(%acc: tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  return %result: tensor<1x1x1x1xf32>
}
func.func @conv2d_accumulate_1_1_16_16_times_2_2_1_dtype_f16_f16_f32(%lhs: tensor<1x1x16x16xf16>, %rhs: tensor<1x1x2x2xf16>, %acc: tensor<1x1x15x15xf32>) -> tensor<1x1x15x15xf32> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<1x1x16x16xf16>, tensor<1x1x2x2xf16>) outs(%acc: tensor<1x1x15x15xf32>) -> tensor<1x1x15x15xf32>
  return %result: tensor<1x1x15x15xf32>
}
func.func @conv2d_accumulate_2_2_32_32_times_3_3_2_dtype_f16_f16_f32(%lhs: tensor<2x2x32x32xf16>, %rhs: tensor<2x2x3x3xf16>, %acc: tensor<2x2x30x30xf32>) -> tensor<2x2x30x30xf32> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x2x32x32xf16>, tensor<2x2x3x3xf16>) outs(%acc: tensor<2x2x30x30xf32>) -> tensor<2x2x30x30xf32>
  return %result: tensor<2x2x30x30xf32>
}
