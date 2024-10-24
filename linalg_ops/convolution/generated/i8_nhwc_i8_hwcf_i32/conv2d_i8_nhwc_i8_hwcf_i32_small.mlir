func.func @conv2d_accumulate_1_1_1_1_times_1_1_1_dtype_i8_i8_i32(%lhs: tensor<1x1x1x1xi8>, %rhs: tensor<1x1x1x1xi8>, %acc: tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<1x1x1x1xi8>, tensor<1x1x1x1xi8>) outs(%acc: tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  return %result: tensor<1x1x1x1xi32>
}
func.func @conv2d_accumulate_1_1_16_16_times_2_2_1_dtype_i8_i8_i32(%lhs: tensor<1x16x16x1xi8>, %rhs: tensor<2x2x1x1xi8>, %acc: tensor<1x15x15x1xi32>) -> tensor<1x15x15x1xi32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<1x16x16x1xi8>, tensor<2x2x1x1xi8>) outs(%acc: tensor<1x15x15x1xi32>) -> tensor<1x15x15x1xi32>
  return %result: tensor<1x15x15x1xi32>
}
func.func @conv2d_accumulate_2_2_32_32_times_3_3_2_dtype_i8_i8_i32(%lhs: tensor<2x32x32x2xi8>, %rhs: tensor<3x3x2x2xi8>, %acc: tensor<2x30x30x2xi32>) -> tensor<2x30x30x2xi32> {
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x32x32x2xi8>, tensor<3x3x2x2xi8>) outs(%acc: tensor<2x30x30x2xi32>) -> tensor<2x30x30x2xi32>
  return %result: tensor<2x30x30x2xi32>
}
