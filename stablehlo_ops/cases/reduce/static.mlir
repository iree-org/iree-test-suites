module {
  func.func @main(%arg0: tensor<2x3xi32>, %arg1: tensor<i32>) -> tensor<2xi32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}

