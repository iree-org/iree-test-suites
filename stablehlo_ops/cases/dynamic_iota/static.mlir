module {
  func.func @main(%arg0: tensor<2xi64>) -> tensor<2x3xi32> {
    %0 = stablehlo.dynamic_iota %arg0, dim = 1 : (tensor<2xi64>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
}

