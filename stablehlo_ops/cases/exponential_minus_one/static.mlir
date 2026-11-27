module {
  func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.exponential_minus_one %arg0 : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}

