module {
  func.func @main(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<?x?xi32>) -> tensor<?x?xi32>
    return %0 : tensor<?x?xi32>
  }
}

