module {
  func.func @main(%input: tensor<3xi32>) -> tensor<3xi32> {
    %result = stablehlo.composite "test.double" %input {decomposition = @double} : (tensor<3xi32>) -> tensor<3xi32>
    return %result : tensor<3xi32>
  }

  func.func private @double(%input: tensor<3xi32>) -> tensor<3xi32> {
    %result = stablehlo.add %input, %input : tensor<3xi32>
    return %result : tensor<3xi32>
  }
}
