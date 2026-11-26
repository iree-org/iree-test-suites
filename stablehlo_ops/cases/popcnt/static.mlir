module {
  func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    %0 = stablehlo.popcnt %arg0 : tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}

