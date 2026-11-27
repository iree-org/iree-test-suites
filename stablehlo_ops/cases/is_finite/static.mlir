module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xi1> {
    %0 = stablehlo.is_finite %arg0 : (tensor<4xf32>) -> tensor<4xi1>
    return %0 : tensor<4xi1>
  }
}

