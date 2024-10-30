func.func @conv2d_accumulate_2_2_32_32_times_3_3_2_dtype_i8_i8_i32(%lhs: tensor<2x2x32x32xi8>, %rhs: tensor<2x2x3x3xi8>, %acc: tensor<2x2x30x30xi32>) -> tensor<2x2x30x30xi32> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x2x32x32xi8>, tensor<2x2x3x3xi8>) outs(%acc: tensor<2x2x30x30xi32>) -> tensor<2x2x30x30xi32>
  return %result: tensor<2x2x30x30xi32>
}
func.func @conv2d_accumulate_2_2_32_32_times_3_3_64_dtype_i8_i8_i32(%lhs: tensor<2x2x32x32xi8>, %rhs: tensor<64x2x3x3xi8>, %acc: tensor<2x64x30x30xi32>) -> tensor<2x64x30x30xi32> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x2x32x32xi8>, tensor<64x2x3x3xi8>) outs(%acc: tensor<2x64x30x30xi32>) -> tensor<2x64x30x30xi32>
  return %result: tensor<2x64x30x30xi32>
}
func.func @conv2d_accumulate_2_16_32_32_times_3_3_64_dtype_i8_i8_i32(%lhs: tensor<2x16x32x32xi8>, %rhs: tensor<64x16x3x3xi8>, %acc: tensor<2x64x30x30xi32>) -> tensor<2x64x30x30xi32> {
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} ins(%lhs, %rhs: tensor<2x16x32x32xi8>, tensor<64x16x3x3xi8>) outs(%acc: tensor<2x64x30x30xi32>) -> tensor<2x64x30x30xi32>
  return %result: tensor<2x64x30x30xi32>
}
