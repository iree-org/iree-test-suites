module {
  func.func @main(%arg0: tensor<?xf32>) -> tensor<?xi1> {
    %0 = stablehlo.is_finite %arg0 : (tensor<?xf32>) -> tensor<?xi1>
    return %0 : tensor<?xi1>
  }
}

