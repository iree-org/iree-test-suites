module {
  func.func @main(%arg0: tensor<5xf32>) -> tensor<5xf32> {
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<5xf32>) -> tensor<5x!quant.uniform<i8:f32, 5.000000e-01>>
    %1 = stablehlo.uniform_dequantize %0 : (tensor<5x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<5xf32>
    return %1 : tensor<5xf32>
  }
}

