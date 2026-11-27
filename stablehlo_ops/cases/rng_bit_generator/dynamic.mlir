module {
  func.func @main(%arg0: tensor<?xui64>) -> (tensor<?xui64>, tensor<4xui32>) {
    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  THREE_FRY : (tensor<?xui64>) -> (tensor<?xui64>, tensor<4xui32>)
    return %output_state, %output : tensor<?xui64>, tensor<4xui32>
  }
}

