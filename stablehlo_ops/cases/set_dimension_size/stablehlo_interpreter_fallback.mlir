module {
  func.func @main(%input: tensor<3xi32>, %size: tensor<i32>) -> tensor<2xi32> {
    // The manifest supplies runtime size 2, so the expected result is the first two elements.
    %prefix = stablehlo.slice %input [0:2] : (tensor<3xi32>) -> tensor<2xi32>
    return %prefix : tensor<2xi32>
  }
}
