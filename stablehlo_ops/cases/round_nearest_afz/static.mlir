module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.round_nearest_afz %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

