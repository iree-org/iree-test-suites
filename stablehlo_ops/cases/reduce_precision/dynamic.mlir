//module {
//  func.func @main(%arg0: tensor<?xf32>) -> tensor<?xf32> {
//    %0 = stablehlo.reduce_precision %arg0, format=e5m10 : tensor<?xf32>
//    return %0 : tensor<?xf32>
//  }
//}

module {
  func.func @main(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %output = "stablehlo.reduce_precision"(%arg0) {
      exponent_bits = 5 : i32,
      mantissa_bits = 10 : i32
    } : (tensor<?xf32>) -> tensor<?xf32>
    return %output: tensor<?xf32>
  }
}
