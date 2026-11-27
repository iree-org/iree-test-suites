module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.cbrt %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

