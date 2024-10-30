func.func @conv2d_accumulate_2_4_128_128_times_3_3_8_dtype_i8_i8_i32(%lhs: tensor<2x128x128x4xi8>, %rhs: tensor<3x3x4x8xi8>, %acc: tensor<2x126x126x8xi32>) -> tensor<2x126x126x8xi32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x128x128x4xi8>, tensor<3x3x4x8xi8>) outs(%acc: tensor<2x126x126x8xi32>) -> tensor<2x126x126x8xi32>
  return %result: tensor<2x126x126x8xi32>
}
func.func @conv2d_accumulate_2_3_128_128_times_3_3_12_dtype_i8_i8_i32(%lhs: tensor<2x128x128x3xi8>, %rhs: tensor<3x3x3x12xi8>, %acc: tensor<2x126x126x12xi32>) -> tensor<2x126x126x12xi32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x128x128x3xi8>, tensor<3x3x3x12xi8>) outs(%acc: tensor<2x126x126x12xi32>) -> tensor<2x126x126x12xi32>
  return %result: tensor<2x126x126x12xi32>
}
