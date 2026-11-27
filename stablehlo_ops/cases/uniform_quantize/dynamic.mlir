module {
  func.func @main(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<?xf32>) -> tensor<?x!quant.uniform<i8:f32, 5.000000e-01>>
    %1 = stablehlo.uniform_dequantize %0 : (tensor<?x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<?xf32>
    return %1 : tensor<?xf32>
  }
}

