module {
  func.func @main(%arg0: tensor<2x3xi32>, %arg1: tensor<1xi64>) -> tensor<?xi32> {
    %0 = stablehlo.dynamic_reshape %arg0, %arg1 : (tensor<2x3xi32>, tensor<1xi64>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

