func.func @matmul_accumulate_512x128xf16_times_128x512xf16_into_512x512xf32(%lhs: tensor<512x128xf16>, %rhs: tensor<128x512xf16>, %acc: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xf16>, tensor<128x512xf16>) outs(%acc: tensor<512x512xf32>) -> tensor<512x512xf32>

  return %result: tensor<512x512xf32>
}

func.func @matmul_512x128xf16_times_128x512xf16_into_512x512xf32(%lhs: tensor<512x128xf16>, %rhs: tensor<128x512xf16>) -> tensor<512x512xf32> {
  %init_acc = tensor.empty() : tensor<512x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<512x512xf32>) -> tensor<512x512xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xf16>, tensor<128x512xf16>) outs(%acc: tensor<512x512xf32>) -> tensor<512x512xf32>
  return %result: tensor<512x512xf32>
}

