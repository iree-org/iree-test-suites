func.func @matmul_accumulate_512x128xi8_times_128x512xi8_into_512x512xi32(%lhs: tensor<512x128xi8>, %rhs: tensor<128x512xi8>, %acc: tensor<512x512xi32>) -> tensor<512x512xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xi8>, tensor<128x512xi8>) outs(%acc: tensor<512x512xi32>) -> tensor<512x512xi32>
  return %result: tensor<512x512xi32>
}

func.func @matmul_512x128xi8_times_128x512xi8_into_512x512xi32(%lhs: tensor<512x128xi8>, %rhs: tensor<128x512xi8>) -> tensor<512x512xi32> {
  %init_acc = tensor.empty() : tensor<512x512xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<512x512xi32>) -> tensor<512x512xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xi8>, tensor<128x512xi8>) outs(%acc: tensor<512x512xi32>) -> tensor<512x512xi32>
  return %result: tensor<512x512xi32>
}

