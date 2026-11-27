module {
  func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %0 = stablehlo.divide %arg0, %arg1 : tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

