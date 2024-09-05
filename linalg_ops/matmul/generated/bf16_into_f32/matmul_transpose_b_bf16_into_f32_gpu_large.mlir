func.func @matmul_457x330xbf16_times_512x330xbf16_into_457x512xf32(%lhs: tensor<457x330xbf16>, %rhs: tensor<512x330xbf16>) -> tensor<457x512xf32> {
  %init_acc = tensor.empty() : tensor<457x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<457x512xf32>) -> tensor<457x512xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<457x330xbf16>, tensor<512x330xbf16>) outs(%acc: tensor<457x512xf32>) -> tensor<457x512xf32>
  return %result: tensor<457x512xf32>
}

func.func @matmul_457x330xbf16_times_514x330xbf16_into_457x514xf32(%lhs: tensor<457x330xbf16>, %rhs: tensor<514x330xbf16>) -> tensor<457x514xf32> {
  %init_acc = tensor.empty() : tensor<457x514xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<457x514xf32>) -> tensor<457x514xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<457x330xbf16>, tensor<514x330xbf16>) outs(%acc: tensor<457x514xf32>) -> tensor<457x514xf32>
  return %result: tensor<457x514xf32>
}

func.func @matmul_438x330xbf16_times_514x330xbf16_into_438x514xf32(%lhs: tensor<438x330xbf16>, %rhs: tensor<514x330xbf16>) -> tensor<438x514xf32> {
  %init_acc = tensor.empty() : tensor<438x514xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<438x514xf32>) -> tensor<438x514xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<438x330xbf16>, tensor<514x330xbf16>) outs(%acc: tensor<438x514xf32>) -> tensor<438x514xf32>
  return %result: tensor<438x514xf32>
}

func.func @matmul_540x332xbf16_times_516x332xbf16_into_540x516xf32(%lhs: tensor<540x332xbf16>, %rhs: tensor<516x332xbf16>) -> tensor<540x516xf32> {
  %init_acc = tensor.empty() : tensor<540x516xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<540x516xf32>) -> tensor<540x516xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<540x332xbf16>, tensor<516x332xbf16>) outs(%acc: tensor<540x516xf32>) -> tensor<540x516xf32>
  return %result: tensor<540x516xf32>
}

func.func @matmul_1000x4xbf16_times_512x4xbf16_into_1000x512xf32(%lhs: tensor<1000x4xbf16>, %rhs: tensor<512x4xbf16>) -> tensor<1000x512xf32> {
  %init_acc = tensor.empty() : tensor<1000x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<1000x512xf32>) -> tensor<1000x512xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1000x4xbf16>, tensor<512x4xbf16>) outs(%acc: tensor<1000x512xf32>) -> tensor<1000x512xf32>
  return %result: tensor<1000x512xf32>
}

func.func @matmul_4x1000xbf16_times_512x1000xbf16_into_4x512xf32(%lhs: tensor<4x1000xbf16>, %rhs: tensor<512x1000xbf16>) -> tensor<4x512xf32> {
  %init_acc = tensor.empty() : tensor<4x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<4x512xf32>) -> tensor<4x512xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<4x1000xbf16>, tensor<512x1000xbf16>) outs(%acc: tensor<4x512xf32>) -> tensor<4x512xf32>
  return %result: tensor<4x512xf32>
}

func.func @matmul_512x1000xbf16_times_4x1000xbf16_into_512x4xf32(%lhs: tensor<512x1000xbf16>, %rhs: tensor<4x1000xbf16>) -> tensor<512x4xf32> {
  %init_acc = tensor.empty() : tensor<512x4xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<512x4xf32>) -> tensor<512x4xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<512x1000xbf16>, tensor<4x1000xbf16>) outs(%acc: tensor<512x4xf32>) -> tensor<512x4xf32>
  return %result: tensor<512x4xf32>
}

func.func @matmul_512x128xbf16_times_500x128xbf16_into_512x500xf32(%lhs: tensor<512x128xbf16>, %rhs: tensor<500x128xbf16>) -> tensor<512x500xf32> {
  %init_acc = tensor.empty() : tensor<512x500xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<512x500xf32>) -> tensor<512x500xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<512x128xbf16>, tensor<500x128xbf16>) outs(%acc: tensor<512x500xf32>) -> tensor<512x500xf32>
  return %result: tensor<512x500xf32>
}

func.func @matmul_457x160xbf16_times_512x160xbf16_into_457x512xf32(%lhs: tensor<457x160xbf16>, %rhs: tensor<512x160xbf16>) -> tensor<457x512xf32> {
  %init_acc = tensor.empty() : tensor<457x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<457x512xf32>) -> tensor<457x512xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<457x160xbf16>, tensor<512x160xbf16>) outs(%acc: tensor<457x512xf32>) -> tensor<457x512xf32>
  return %result: tensor<457x512xf32>
}

func.func @matmul_512x330xbf16_times_512x330xbf16_into_512x512xf32(%lhs: tensor<512x330xbf16>, %rhs: tensor<512x330xbf16>) -> tensor<512x512xf32> {
  %init_acc = tensor.empty() : tensor<512x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<512x512xf32>) -> tensor<512x512xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<512x330xbf16>, tensor<512x330xbf16>) outs(%acc: tensor<512x512xf32>) -> tensor<512x512xf32>
  return %result: tensor<512x512xf32>
}

