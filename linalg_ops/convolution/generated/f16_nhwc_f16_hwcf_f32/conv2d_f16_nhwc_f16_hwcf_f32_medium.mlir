func.func @conv2d_accumulate_2_2_32_32_times_3_3_2_dtype_f16_f16_f32(%lhs: tensor<2x32x32x2xf16>, %rhs: tensor<3x3x2x2xf16>, %acc: tensor<2x30x30x2xf32>) -> tensor<2x30x30x2xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x32x32x2xf16>, tensor<3x3x2x2xf16>) outs(%acc: tensor<2x30x30x2xf32>) -> tensor<2x30x30x2xf32>
  return %result: tensor<2x30x30x2xf32>
}
func.func @conv2d_accumulate_2_2_32_32_times_3_3_64_dtype_f16_f16_f32(%lhs: tensor<2x32x32x2xf16>, %rhs: tensor<3x3x2x64xf16>, %acc: tensor<2x30x30x64xf32>) -> tensor<2x30x30x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x32x32x2xf16>, tensor<3x3x2x64xf16>) outs(%acc: tensor<2x30x30x64xf32>) -> tensor<2x30x30x64xf32>
  return %result: tensor<2x30x30x64xf32>
}
func.func @conv2d_accumulate_2_32_32_32_times_3_3_64_dtype_f16_f16_f32(%lhs: tensor<2x32x32x32xf16>, %rhs: tensor<3x3x32x64xf16>, %acc: tensor<2x30x30x64xf32>) -> tensor<2x30x30x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x32x32x32xf16>, tensor<3x3x32x64xf16>) outs(%acc: tensor<2x30x30x64xf32>) -> tensor<2x30x30x64xf32>
  return %result: tensor<2x30x30x64xf32>
}
