module {
  func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = stablehlo.popcnt %arg0 : tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

