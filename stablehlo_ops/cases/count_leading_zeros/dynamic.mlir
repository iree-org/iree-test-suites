module {
  func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = stablehlo.count_leading_zeros %arg0 : tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

