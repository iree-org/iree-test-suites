func.func @matmul_457x330xf8E4M3FNUZ_times_330x512xf8E4M3FNUZ_into_457x512xf32(%lhs: tensor<457x330xf32>, %rhs: tensor<330x512xf32>) -> tensor<457x512xf32> {
  %init_acc = tensor.empty() : tensor<457x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<457x512xf32>) -> tensor<457x512xf32>
  %lhs_casted = arith.truncf %lhs: tensor<457x330xf32> to tensor<457x330xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<330x512xf32> to tensor<330x512xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<457x330xf8E4M3FNUZ>, tensor<330x512xf8E4M3FNUZ>) outs(%acc: tensor<457x512xf32>) -> tensor<457x512xf32>  return %result: tensor<457x512xf32>
}

func.func @matmul_457x330xf8E4M3FNUZ_times_330x514xf8E4M3FNUZ_into_457x514xf32(%lhs: tensor<457x330xf32>, %rhs: tensor<330x514xf32>) -> tensor<457x514xf32> {
  %init_acc = tensor.empty() : tensor<457x514xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<457x514xf32>) -> tensor<457x514xf32>
  %lhs_casted = arith.truncf %lhs: tensor<457x330xf32> to tensor<457x330xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<330x514xf32> to tensor<330x514xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<457x330xf8E4M3FNUZ>, tensor<330x514xf8E4M3FNUZ>) outs(%acc: tensor<457x514xf32>) -> tensor<457x514xf32>  return %result: tensor<457x514xf32>
}

func.func @matmul_438x330xf8E4M3FNUZ_times_330x514xf8E4M3FNUZ_into_438x514xf32(%lhs: tensor<438x330xf32>, %rhs: tensor<330x514xf32>) -> tensor<438x514xf32> {
  %init_acc = tensor.empty() : tensor<438x514xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<438x514xf32>) -> tensor<438x514xf32>
  %lhs_casted = arith.truncf %lhs: tensor<438x330xf32> to tensor<438x330xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<330x514xf32> to tensor<330x514xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<438x330xf8E4M3FNUZ>, tensor<330x514xf8E4M3FNUZ>) outs(%acc: tensor<438x514xf32>) -> tensor<438x514xf32>  return %result: tensor<438x514xf32>
}

func.func @matmul_540x332xf8E4M3FNUZ_times_332x516xf8E4M3FNUZ_into_540x516xf32(%lhs: tensor<540x332xf32>, %rhs: tensor<332x516xf32>) -> tensor<540x516xf32> {
  %init_acc = tensor.empty() : tensor<540x516xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<540x516xf32>) -> tensor<540x516xf32>
  %lhs_casted = arith.truncf %lhs: tensor<540x332xf32> to tensor<540x332xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<332x516xf32> to tensor<332x516xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<540x332xf8E4M3FNUZ>, tensor<332x516xf8E4M3FNUZ>) outs(%acc: tensor<540x516xf32>) -> tensor<540x516xf32>  return %result: tensor<540x516xf32>
}

func.func @matmul_1000x4xf8E4M3FNUZ_times_4x512xf8E4M3FNUZ_into_1000x512xf32(%lhs: tensor<1000x4xf32>, %rhs: tensor<4x512xf32>) -> tensor<1000x512xf32> {
  %init_acc = tensor.empty() : tensor<1000x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<1000x512xf32>) -> tensor<1000x512xf32>
  %lhs_casted = arith.truncf %lhs: tensor<1000x4xf32> to tensor<1000x4xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<4x512xf32> to tensor<4x512xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<1000x4xf8E4M3FNUZ>, tensor<4x512xf8E4M3FNUZ>) outs(%acc: tensor<1000x512xf32>) -> tensor<1000x512xf32>  return %result: tensor<1000x512xf32>
}

func.func @matmul_4x1000xf8E4M3FNUZ_times_1000x512xf8E4M3FNUZ_into_4x512xf32(%lhs: tensor<4x1000xf32>, %rhs: tensor<1000x512xf32>) -> tensor<4x512xf32> {
  %init_acc = tensor.empty() : tensor<4x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<4x512xf32>) -> tensor<4x512xf32>
  %lhs_casted = arith.truncf %lhs: tensor<4x1000xf32> to tensor<4x1000xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<1000x512xf32> to tensor<1000x512xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<4x1000xf8E4M3FNUZ>, tensor<1000x512xf8E4M3FNUZ>) outs(%acc: tensor<4x512xf32>) -> tensor<4x512xf32>  return %result: tensor<4x512xf32>
}

func.func @matmul_512x1000xf8E4M3FNUZ_times_1000x4xf8E4M3FNUZ_into_512x4xf32(%lhs: tensor<512x1000xf32>, %rhs: tensor<1000x4xf32>) -> tensor<512x4xf32> {
  %init_acc = tensor.empty() : tensor<512x4xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<512x4xf32>) -> tensor<512x4xf32>
  %lhs_casted = arith.truncf %lhs: tensor<512x1000xf32> to tensor<512x1000xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<1000x4xf32> to tensor<1000x4xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<512x1000xf8E4M3FNUZ>, tensor<1000x4xf8E4M3FNUZ>) outs(%acc: tensor<512x4xf32>) -> tensor<512x4xf32>  return %result: tensor<512x4xf32>
}

func.func @matmul_512x128xf8E4M3FNUZ_times_128x500xf8E4M3FNUZ_into_512x500xf32(%lhs: tensor<512x128xf32>, %rhs: tensor<128x500xf32>) -> tensor<512x500xf32> {
  %init_acc = tensor.empty() : tensor<512x500xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<512x500xf32>) -> tensor<512x500xf32>
  %lhs_casted = arith.truncf %lhs: tensor<512x128xf32> to tensor<512x128xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<128x500xf32> to tensor<128x500xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<512x128xf8E4M3FNUZ>, tensor<128x500xf8E4M3FNUZ>) outs(%acc: tensor<512x500xf32>) -> tensor<512x500xf32>  return %result: tensor<512x500xf32>
}

func.func @matmul_457x160xf8E4M3FNUZ_times_160x512xf8E4M3FNUZ_into_457x512xf32(%lhs: tensor<457x160xf32>, %rhs: tensor<160x512xf32>) -> tensor<457x512xf32> {
  %init_acc = tensor.empty() : tensor<457x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<457x512xf32>) -> tensor<457x512xf32>
  %lhs_casted = arith.truncf %lhs: tensor<457x160xf32> to tensor<457x160xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<160x512xf32> to tensor<160x512xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<457x160xf8E4M3FNUZ>, tensor<160x512xf8E4M3FNUZ>) outs(%acc: tensor<457x512xf32>) -> tensor<457x512xf32>  return %result: tensor<457x512xf32>
}

func.func @matmul_512x330xf8E4M3FNUZ_times_330x512xf8E4M3FNUZ_into_512x512xf32(%lhs: tensor<512x330xf32>, %rhs: tensor<330x512xf32>) -> tensor<512x512xf32> {
  %init_acc = tensor.empty() : tensor<512x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<512x512xf32>) -> tensor<512x512xf32>
  %lhs_casted = arith.truncf %lhs: tensor<512x330xf32> to tensor<512x330xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<330x512xf32> to tensor<330x512xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<512x330xf8E4M3FNUZ>, tensor<330x512xf8E4M3FNUZ>) outs(%acc: tensor<512x512xf32>) -> tensor<512x512xf32>  return %result: tensor<512x512xf32>
}

