module {
  func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.logistic %arg0 : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}

