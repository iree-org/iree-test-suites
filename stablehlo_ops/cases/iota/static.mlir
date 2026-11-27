module {
  func.func @main(%input: tensor<2x3xi32>) -> tensor<4x3xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<2x3xi32>
    %joined = stablehlo.concatenate %input, %0, dim = 0 : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<4x3xi32>
    return %joined : tensor<4x3xi32>
  }
}
