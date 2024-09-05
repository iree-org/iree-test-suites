func.func @matmul_accumulate_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf32(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<?x?xf16>, tensor<?x?xf16>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}

func.func @matmul_accumulate_512x128xf16_times_128x512xf16_into_512x512xf32(%lhs: tensor<512x128xf16>, %rhs: tensor<128x512xf16>, %acc: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xf16>, tensor<128x512xf16>) outs(%acc: tensor<512x512xf32>) -> tensor<512x512xf32>
  return %result: tensor<512x512xf32>
}

func.func @matmul_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf32(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc_dim0 = tensor.dim %lhs, %c0 : tensor<?x?xf16>
  %acc_dim1 = tensor.dim %rhs, %c1 : tensor<?x?xf16>
  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : tensor<?x?xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<?x?xf16>, tensor<?x?xf16>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}

func.func @matmul_512x128xf16_times_128x512xf16_into_512x512xf32(%lhs: tensor<512x128xf16>, %rhs: tensor<128x512xf16>) -> tensor<512x512xf32> {
  %init_acc = tensor.empty() : tensor<512x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<512x512xf32>) -> tensor<512x512xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xf16>, tensor<128x512xf16>) outs(%acc: tensor<512x512xf32>) -> tensor<512x512xf32>
  return %result: tensor<512x512xf32>
}

func.func @matmul_1000x4xf16_times_4x512xf16_into_1000x512xf32(%lhs: tensor<1000x4xf16>, %rhs: tensor<4x512xf16>) -> tensor<1000x512xf32> {
  %init_acc = tensor.empty() : tensor<1000x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<1000x512xf32>) -> tensor<1000x512xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1000x4xf16>, tensor<4x512xf16>) outs(%acc: tensor<1000x512xf32>) -> tensor<1000x512xf32>
  return %result: tensor<1000x512xf32>
}

func.func @matmul_4x1000xf16_times_1000x512xf16_into_4x512xf32(%lhs: tensor<4x1000xf16>, %rhs: tensor<1000x512xf16>) -> tensor<4x512xf32> {
  %init_acc = tensor.empty() : tensor<4x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<4x512xf32>) -> tensor<4x512xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<4x1000xf16>, tensor<1000x512xf16>) outs(%acc: tensor<4x512xf32>) -> tensor<4x512xf32>
  return %result: tensor<4x512xf32>
}

func.func @matmul_512x1000xf16_times_1000x4xf16_into_512x4xf32(%lhs: tensor<512x1000xf16>, %rhs: tensor<1000x4xf16>) -> tensor<512x4xf32> {
  %init_acc = tensor.empty() : tensor<512x4xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<512x4xf32>) -> tensor<512x4xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x1000xf16>, tensor<1000x4xf16>) outs(%acc: tensor<512x4xf32>) -> tensor<512x4xf32>
  return %result: tensor<512x4xf32>
}

func.func @matmul_512x128xf16_times_128x500xf16_into_512x500xf32(%lhs: tensor<512x128xf16>, %rhs: tensor<128x500xf16>) -> tensor<512x500xf32> {
  %init_acc = tensor.empty() : tensor<512x500xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<512x500xf32>) -> tensor<512x500xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xf16>, tensor<128x500xf16>) outs(%acc: tensor<512x500xf32>) -> tensor<512x500xf32>
  return %result: tensor<512x500xf32>
}

func.func @matmul_accumulate_123x456xf16_times_456x789xf16_into_123x789xf32(%lhs: tensor<123x456xf16>, %rhs: tensor<456x789xf16>, %acc: tensor<123x789xf32>) -> tensor<123x789xf32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<123x456xf16>, tensor<456x789xf16>) outs(%acc: tensor<123x789xf32>) -> tensor<123x789xf32>
  return %result: tensor<123x789xf32>
}

func.func @matmul_457x330xf16_times_330x512xf16_into_457x512xf32(%lhs: tensor<457x330xf16>, %rhs: tensor<330x512xf16>) -> tensor<457x512xf32> {
  %init_acc = tensor.empty() : tensor<457x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<457x512xf32>) -> tensor<457x512xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<457x330xf16>, tensor<330x512xf16>) outs(%acc: tensor<457x512xf32>) -> tensor<457x512xf32>
  return %result: tensor<457x512xf32>
}

func.func @matmul_457x330xf16_times_330x514xf16_into_457x514xf32(%lhs: tensor<457x330xf16>, %rhs: tensor<330x514xf16>) -> tensor<457x514xf32> {
  %init_acc = tensor.empty() : tensor<457x514xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<457x514xf32>) -> tensor<457x514xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<457x330xf16>, tensor<330x514xf16>) outs(%acc: tensor<457x514xf32>) -> tensor<457x514xf32>
  return %result: tensor<457x514xf32>
}

func.func @matmul_438x330xf16_times_330x514xf16_into_438x514xf32(%lhs: tensor<438x330xf16>, %rhs: tensor<330x514xf16>) -> tensor<438x514xf32> {
  %init_acc = tensor.empty() : tensor<438x514xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<438x514xf32>) -> tensor<438x514xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<438x330xf16>, tensor<330x514xf16>) outs(%acc: tensor<438x514xf32>) -> tensor<438x514xf32>
  return %result: tensor<438x514xf32>
}

func.func @matmul_540x332xf16_times_332x516xf16_into_540x516xf32(%lhs: tensor<540x332xf16>, %rhs: tensor<332x516xf16>) -> tensor<540x516xf32> {
  %init_acc = tensor.empty() : tensor<540x516xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<540x516xf32>) -> tensor<540x516xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<540x332xf16>, tensor<332x516xf16>) outs(%acc: tensor<540x516xf32>) -> tensor<540x516xf32>
  return %result: tensor<540x516xf32>
}

func.func @matmul_654x321xf16_times_321x234xf16_into_654x234xf32(%lhs: tensor<654x321xf16>, %rhs: tensor<321x234xf16>) -> tensor<654x234xf32> {
  %init_acc = tensor.empty() : tensor<654x234xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<654x234xf32>) -> tensor<654x234xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<654x321xf16>, tensor<321x234xf16>) outs(%acc: tensor<654x234xf32>) -> tensor<654x234xf32>
  return %result: tensor<654x234xf32>
}

func.func @matmul_457x160xf16_times_160x512xf16_into_457x512xf32(%lhs: tensor<457x160xf16>, %rhs: tensor<160x512xf16>) -> tensor<457x512xf32> {
  %init_acc = tensor.empty() : tensor<457x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<457x512xf32>) -> tensor<457x512xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<457x160xf16>, tensor<160x512xf16>) outs(%acc: tensor<457x512xf32>) -> tensor<457x512xf32>
  return %result: tensor<457x512xf32>
}

func.func @matmul_512x330xf16_times_330x512xf16_into_512x512xf32(%lhs: tensor<512x330xf16>, %rhs: tensor<330x512xf16>) -> tensor<512x512xf32> {
  %init_acc = tensor.empty() : tensor<512x512xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<512x512xf32>) -> tensor<512x512xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x330xf16>, tensor<330x512xf16>) outs(%acc: tensor<512x512xf32>) -> tensor<512x512xf32>
  return %result: tensor<512x512xf32>
}

func.func @matmul_accumulate_1x1000xf16_times_1000x1000xf16_into_1x1000xf32(%lhs: tensor<1x1000xf16>, %rhs: tensor<1000x1000xf16>, %acc: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1x1000xf16>, tensor<1000x1000xf16>) outs(%acc: tensor<1x1000xf32>) -> tensor<1x1000xf32>
  return %result: tensor<1x1000xf32>
}

func.func @matmul_accumulate_1000x1000xf16_times_1000x1xf16_into_1000x1xf32(%lhs: tensor<1000x1000xf16>, %rhs: tensor<1000x1xf16>, %acc: tensor<1000x1xf32>) -> tensor<1000x1xf32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1000x1000xf16>, tensor<1000x1xf16>) outs(%acc: tensor<1000x1xf32>) -> tensor<1000x1xf32>
  return %result: tensor<1000x1xf32>
}

func.func @matmul_1000x1000xf16_times_1000x1xf16_into_1000x1xf32(%lhs: tensor<1000x1000xf16>, %rhs: tensor<1000x1xf16>) -> tensor<1000x1xf32> {
  %init_acc = tensor.empty() : tensor<1000x1xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<1000x1xf32>) -> tensor<1000x1xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1000x1000xf16>, tensor<1000x1xf16>) outs(%acc: tensor<1000x1xf32>) -> tensor<1000x1xf32>
  return %result: tensor<1000x1xf32>
}

