module {
  func.func @main(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi32> {
    %0 = stablehlo.xor %arg0, %arg1 : tensor<3xi32>
    return %0 : tensor<3xi32>
  }
}

