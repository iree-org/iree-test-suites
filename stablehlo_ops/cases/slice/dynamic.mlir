module {
  func.func @main(%arg0: tensor<?x3xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.slice %arg0 [0:2, 0:3:2] : (tensor<?x3xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}

