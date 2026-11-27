module {
  func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.reduce_precision %arg0, format = e5m10 : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}

