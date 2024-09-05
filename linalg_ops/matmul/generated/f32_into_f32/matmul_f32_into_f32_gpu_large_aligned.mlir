func.func @matmul_accumulate_512x128xf32_times_128x512xf32_into_512x512xf32(%lhs: tensor<512x128xf32>, %rhs: tensor<128x512xf32>, %acc: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xf32>, tensor<128x512xf32>) outs(%acc: tensor<512x512xf32>) -> tensor<512x512xf32>

  return %result: tensor<512x512xf32>
}

func.func @matmul_512x128xf32_times_128x512xf32_into_512x512xf32(%lhs: tensor<512x128xf32>, %rhs: tensor<128x512xf32>) -> tensor<512x512xf32> {
  %init_acc = tensor.empty() : tensor<512x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<512x512xf32>) -> tensor<512x512xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xf32>, tensor<128x512xf32>) outs(%acc: tensor<512x512xf32>) -> tensor<512x512xf32>
  return %result: tensor<512x512xf32>
}

