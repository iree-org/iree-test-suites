module {
  func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    %0 = stablehlo.count_leading_zeros %arg0 : tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}

