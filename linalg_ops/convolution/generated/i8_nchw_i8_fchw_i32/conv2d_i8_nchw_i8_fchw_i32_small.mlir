func.func @conv2d_accumulate_1_1_1_1_times_1_1_1_dtype_i8_i8_i32(%lhs: tensor<1x1x1x1xi8>, %rhs: tensor<1x1x1x1xi8>, %acc: tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<1x1x1x1xi8>, tensor<1x1x1x1xi8>) outs(%acc: tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
  return %result: tensor<1x1x1x1xi32>
}
func.func @conv2d_accumulate_1_1_16_16_times_2_2_1_dtype_i8_i8_i32(%lhs: tensor<1x1x16x16xi8>, %rhs: tensor<1x1x2x2xi8>, %acc: tensor<1x1x15x15xi32>) -> tensor<1x1x15x15xi32> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<1x1x16x16xi8>, tensor<1x1x2x2xi8>) outs(%acc: tensor<1x1x15x15xi32>) -> tensor<1x1x15x15xi32>
  return %result: tensor<1x1x15x15xi32>
}
func.func @conv2d_accumulate_2_2_32_32_times_3_3_2_dtype_i8_i8_i32(%lhs: tensor<2x2x32x32xi8>, %rhs: tensor<2x2x3x3xi8>, %acc: tensor<2x2x30x30xi32>) -> tensor<2x2x30x30xi32> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x2x32x32xi8>, tensor<2x2x3x3xi8>) outs(%acc: tensor<2x2x30x30xi32>) -> tensor<2x2x30x30xi32>
  return %result: tensor<2x2x30x30xi32>
}
