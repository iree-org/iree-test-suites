module {
  func.func @main(%arg0: tensor<?x3xi32>) -> tensor<6xi32> {
    %0 = stablehlo.reshape %arg0 : (tensor<?x3xi32>) -> tensor<6xi32>
    return %0 : tensor<6xi32>
  }
}

