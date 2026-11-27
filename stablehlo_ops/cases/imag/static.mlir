module {
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.complex %arg0, %arg1 : tensor<2xcomplex<f32>>
    %1 = stablehlo.imag %0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}

