module {
  func.func @main(%arg0: tensor<3xi32>) -> tensor<2x3xi32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
}

