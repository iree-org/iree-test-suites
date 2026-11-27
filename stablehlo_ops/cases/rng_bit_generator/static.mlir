module {
  func.func @main(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>) {
    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    return %output_state, %output : tensor<2xui64>, tensor<4xui32>
  }
}

