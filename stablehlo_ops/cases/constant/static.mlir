module {
  func.func @main(%input: tensor<3xi32>) -> tensor<6xi32> {
    %c = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
    %joined = stablehlo.concatenate %input, %c, dim = 0 : (tensor<3xi32>, tensor<3xi32>) -> tensor<6xi32>
    return %joined : tensor<6xi32>
  }
}
