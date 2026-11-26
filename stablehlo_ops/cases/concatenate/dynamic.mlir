module {
  func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

