module {
  func.func @main(%arg0: tensor<1x3x3x1xf32>, %arg1: tensor<2x2x1x1xf32>) -> tensor<1x2x2x1xf32> {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x1xf32>, tensor<2x2x1x1xf32>) -> tensor<1x2x2x1xf32>
    return %0 : tensor<1x2x2x1xf32>
  }
}

