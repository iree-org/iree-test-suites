module {
  func.func @main(%input: tensor<?xi32>) -> tensor<?xi32> {
    %result = stablehlo.composite "test.double" %input {decomposition = @double} : (tensor<?xi32>) -> tensor<?xi32>
    return %result : tensor<?xi32>
  }

  func.func private @double(%input: tensor<?xi32>) -> tensor<?xi32> {
    %result = stablehlo.add %input, %input : tensor<?xi32>
    return %result : tensor<?xi32>
  }
}
