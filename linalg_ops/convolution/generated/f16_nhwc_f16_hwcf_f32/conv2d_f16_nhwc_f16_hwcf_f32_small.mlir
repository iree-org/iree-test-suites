func.func @conv2d_accumulate_1_1_1_1_times_1_1_1_dtype_f16_f16_f32(%lhs: tensor<1x1x1x1xf16>, %rhs: tensor<1x1x1x1xf16>, %acc: tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>) outs(%acc: tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  return %result: tensor<1x1x1x1xf32>
}
func.func @conv2d_accumulate_1_1_16_16_times_2_2_1_dtype_f16_f16_f32(%lhs: tensor<1x16x16x1xf16>, %rhs: tensor<2x2x1x1xf16>, %acc: tensor<1x15x15x1xf32>) -> tensor<1x15x15x1xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<1x16x16x1xf16>, tensor<2x2x1x1xf16>) outs(%acc: tensor<1x15x15x1xf32>) -> tensor<1x15x15x1xf32>
  return %result: tensor<1x15x15x1xf32>
}
func.func @conv2d_accumulate_2_2_32_32_times_3_3_2_dtype_f16_f16_f32(%lhs: tensor<2x32x32x2xf16>, %rhs: tensor<3x3x2x2xf16>, %acc: tensor<2x30x30x2xf32>) -> tensor<2x30x30x2xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x32x32x2xf16>, tensor<3x3x2x2xf16>) outs(%acc: tensor<2x30x30x2xf32>) -> tensor<2x30x30x2xf32>
  return %result: tensor<2x30x30x2xf32>
}
