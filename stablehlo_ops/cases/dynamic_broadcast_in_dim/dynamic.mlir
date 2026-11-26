module {
  func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<2xi64>) -> tensor<?x3xi32> {
    %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [1] : (tensor<?xi32>, tensor<2xi64>) -> tensor<?x3xi32>
    return %0 : tensor<?x3xi32>
  }
}

