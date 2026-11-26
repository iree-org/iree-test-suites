module {
  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}

