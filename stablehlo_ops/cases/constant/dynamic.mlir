module {
  func.func @main(%input: tensor<?xi32>) -> tensor<?xi32> {
    %c = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
    %joined = stablehlo.concatenate %input, %c, dim = 0 : (tensor<?xi32>, tensor<3xi32>) -> tensor<?xi32>
    return %joined : tensor<?xi32>
  }
}
