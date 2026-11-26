module {
  func.func @main(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
    %0 = stablehlo.reverse %arg0, dims = [1] : tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
}

