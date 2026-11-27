module {
  func.func @main(%arg0: tensor<2x3xi32>, %arg1: tensor<1x2xi32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<2x3xi32> {
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 : (tensor<2x3xi32>, tensor<1x2xi32>, tensor<i32>, tensor<i32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
}

