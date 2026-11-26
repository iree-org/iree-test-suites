module {
  func.func @main(%arg0: tensor<?xf32>) -> tensor<?xi32> {
    %0 = stablehlo.convert %arg0 : (tensor<?xf32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

