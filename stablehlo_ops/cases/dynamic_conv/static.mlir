module {
  func.func @main(%arg0: tensor<1x3x3x1xf32>, %arg1: tensor<2x2x1x1xf32>, %arg2: tensor<2x2xi64>) -> tensor<1x2x2x1xf32> {
    %0 = "stablehlo.dynamic_conv"(%arg0, %arg1, %arg2) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64}> : (tensor<1x3x3x1xf32>, tensor<2x2x1x1xf32>, tensor<2x2xi64>) -> tensor<1x2x2x1xf32>
    return %0 : tensor<1x2x2x1xf32>
  }
}

