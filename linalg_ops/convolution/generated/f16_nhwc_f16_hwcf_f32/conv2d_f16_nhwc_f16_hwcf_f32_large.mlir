func.func @conv2d_accumulate_2_4_128_128_times_3_3_8_dtype_f16_f16_f32(%lhs: tensor<2x128x128x4xf16>, %rhs: tensor<3x3x4x8xf16>, %acc: tensor<2x126x126x8xf32>) -> tensor<2x126x126x8xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x128x128x4xf16>, tensor<3x3x4x8xf16>) outs(%acc: tensor<2x126x126x8xf32>) -> tensor<2x126x126x8xf32>
  return %result: tensor<2x126x126x8xf32>
}
func.func @conv2d_accumulate_2_3_128_128_times_3_3_12_dtype_f16_f16_f32(%lhs: tensor<2x128x128x3xf16>, %rhs: tensor<3x3x3x12xf16>, %acc: tensor<2x126x126x12xf32>) -> tensor<2x126x126x12xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x128x128x3xf16>, tensor<3x3x3x12xf16>) outs(%acc: tensor<2x126x126x12xf32>) -> tensor<2x126x126x12xf32>
  return %result: tensor<2x126x126x12xf32>
}
