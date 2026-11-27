module {
  func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.complex %arg0, %arg1 : tensor<?xcomplex<f32>>
    %1 = stablehlo.real %0 : (tensor<?xcomplex<f32>>) -> tensor<?xf32>
    %2 = stablehlo.imag %0 : (tensor<?xcomplex<f32>>) -> tensor<?xf32>
    return %1, %2 : tensor<?xf32>, tensor<?xf32>
  }
}

