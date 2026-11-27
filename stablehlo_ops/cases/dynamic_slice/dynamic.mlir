module {
  func.func @main(%arg0: tensor<?x3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<1x2xi32> {
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [1, 2] : (tensor<?x3xi32>, tensor<i32>, tensor<i32>) -> tensor<1x2xi32>
    return %0 : tensor<1x2xi32>
  }
}

