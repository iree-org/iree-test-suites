module {
  func.func @main(%arg0: tensor<?x3xi32>) -> tensor<?x3xi32> {
    %0 = stablehlo.reverse %arg0, dims = [1] : tensor<?x3xi32>
    return %0 : tensor<?x3xi32>
  }
}

