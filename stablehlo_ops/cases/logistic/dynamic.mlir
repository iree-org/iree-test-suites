module {
  func.func @main(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %0 = stablehlo.logistic %arg0 : tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

