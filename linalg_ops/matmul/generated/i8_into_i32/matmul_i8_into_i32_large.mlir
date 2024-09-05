func.func @matmul_accumulate_DYNxDYNxi8_times_DYNxDYNxi8_into_DYNxDYNxi32(%lhs: tensor<?x?xi8>, %rhs: tensor<?x?xi8>, %acc: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<?x?xi8>, tensor<?x?xi8>) outs(%acc: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %result: tensor<?x?xi32>
}

func.func @matmul_accumulate_512x128xi8_times_128x512xi8_into_512x512xi32(%lhs: tensor<512x128xi8>, %rhs: tensor<128x512xi8>, %acc: tensor<512x512xi32>) -> tensor<512x512xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xi8>, tensor<128x512xi8>) outs(%acc: tensor<512x512xi32>) -> tensor<512x512xi32>
  return %result: tensor<512x512xi32>
}

func.func @matmul_DYNxDYNxi8_times_DYNxDYNxi8_into_DYNxDYNxi32(%lhs: tensor<?x?xi8>, %rhs: tensor<?x?xi8>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc_dim0 = tensor.dim %lhs, %c0 : tensor<?x?xi8>
  %acc_dim1 = tensor.dim %rhs, %c1 : tensor<?x?xi8>
  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : tensor<?x?xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<?x?xi32>) -> tensor<?x?xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<?x?xi8>, tensor<?x?xi8>) outs(%acc: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %result: tensor<?x?xi32>
}

func.func @matmul_512x128xi8_times_128x512xi8_into_512x512xi32(%lhs: tensor<512x128xi8>, %rhs: tensor<128x512xi8>) -> tensor<512x512xi32> {
  %init_acc = tensor.empty() : tensor<512x512xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<512x512xi32>) -> tensor<512x512xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xi8>, tensor<128x512xi8>) outs(%acc: tensor<512x512xi32>) -> tensor<512x512xi32>
  return %result: tensor<512x512xi32>
}

func.func @matmul_1000x4xi8_times_4x512xi8_into_1000x512xi32(%lhs: tensor<1000x4xi8>, %rhs: tensor<4x512xi8>) -> tensor<1000x512xi32> {
  %init_acc = tensor.empty() : tensor<1000x512xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<1000x512xi32>) -> tensor<1000x512xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1000x4xi8>, tensor<4x512xi8>) outs(%acc: tensor<1000x512xi32>) -> tensor<1000x512xi32>
  return %result: tensor<1000x512xi32>
}

func.func @matmul_4x1000xi8_times_1000x512xi8_into_4x512xi32(%lhs: tensor<4x1000xi8>, %rhs: tensor<1000x512xi8>) -> tensor<4x512xi32> {
  %init_acc = tensor.empty() : tensor<4x512xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<4x512xi32>) -> tensor<4x512xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<4x1000xi8>, tensor<1000x512xi8>) outs(%acc: tensor<4x512xi32>) -> tensor<4x512xi32>
  return %result: tensor<4x512xi32>
}

func.func @matmul_512x1000xi8_times_1000x4xi8_into_512x4xi32(%lhs: tensor<512x1000xi8>, %rhs: tensor<1000x4xi8>) -> tensor<512x4xi32> {
  %init_acc = tensor.empty() : tensor<512x4xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<512x4xi32>) -> tensor<512x4xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x1000xi8>, tensor<1000x4xi8>) outs(%acc: tensor<512x4xi32>) -> tensor<512x4xi32>
  return %result: tensor<512x4xi32>
}

func.func @matmul_512x128xi8_times_128x500xi8_into_512x500xi32(%lhs: tensor<512x128xi8>, %rhs: tensor<128x500xi8>) -> tensor<512x500xi32> {
  %init_acc = tensor.empty() : tensor<512x500xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<512x500xi32>) -> tensor<512x500xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xi8>, tensor<128x500xi8>) outs(%acc: tensor<512x500xi32>) -> tensor<512x500xi32>
  return %result: tensor<512x500xi32>
}

func.func @matmul_457x330xi8_times_330x512xi8_into_457x512xi32(%lhs: tensor<457x330xi8>, %rhs: tensor<330x512xi8>) -> tensor<457x512xi32> {
  %init_acc = tensor.empty() : tensor<457x512xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<457x512xi32>) -> tensor<457x512xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<457x330xi8>, tensor<330x512xi8>) outs(%acc: tensor<457x512xi32>) -> tensor<457x512xi32>
  return %result: tensor<457x512xi32>
}

func.func @matmul_457x330xi8_times_330x514xi8_into_457x514xi32(%lhs: tensor<457x330xi8>, %rhs: tensor<330x514xi8>) -> tensor<457x514xi32> {
  %init_acc = tensor.empty() : tensor<457x514xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<457x514xi32>) -> tensor<457x514xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<457x330xi8>, tensor<330x514xi8>) outs(%acc: tensor<457x514xi32>) -> tensor<457x514xi32>
  return %result: tensor<457x514xi32>
}

func.func @matmul_438x330xi8_times_330x514xi8_into_438x514xi32(%lhs: tensor<438x330xi8>, %rhs: tensor<330x514xi8>) -> tensor<438x514xi32> {
  %init_acc = tensor.empty() : tensor<438x514xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<438x514xi32>) -> tensor<438x514xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<438x330xi8>, tensor<330x514xi8>) outs(%acc: tensor<438x514xi32>) -> tensor<438x514xi32>
  return %result: tensor<438x514xi32>
}

func.func @matmul_540x332xi8_times_332x516xi8_into_540x516xi32(%lhs: tensor<540x332xi8>, %rhs: tensor<332x516xi8>) -> tensor<540x516xi32> {
  %init_acc = tensor.empty() : tensor<540x516xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<540x516xi32>) -> tensor<540x516xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<540x332xi8>, tensor<332x516xi8>) outs(%acc: tensor<540x516xi32>) -> tensor<540x516xi32>
  return %result: tensor<540x516xi32>
}

func.func @matmul_654x321xi8_times_321x234xi8_into_654x234xi32(%lhs: tensor<654x321xi8>, %rhs: tensor<321x234xi8>) -> tensor<654x234xi32> {
  %init_acc = tensor.empty() : tensor<654x234xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<654x234xi32>) -> tensor<654x234xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<654x321xi8>, tensor<321x234xi8>) outs(%acc: tensor<654x234xi32>) -> tensor<654x234xi32>
  return %result: tensor<654x234xi32>
}

func.func @matmul_457x160xi8_times_160x512xi8_into_457x512xi32(%lhs: tensor<457x160xi8>, %rhs: tensor<160x512xi8>) -> tensor<457x512xi32> {
  %init_acc = tensor.empty() : tensor<457x512xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<457x512xi32>) -> tensor<457x512xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<457x160xi8>, tensor<160x512xi8>) outs(%acc: tensor<457x512xi32>) -> tensor<457x512xi32>
  return %result: tensor<457x512xi32>
}

func.func @matmul_512x330xi8_times_330x512xi8_into_512x512xi32(%lhs: tensor<512x330xi8>, %rhs: tensor<330x512xi8>) -> tensor<512x512xi32> {
  %init_acc = tensor.empty() : tensor<512x512xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<512x512xi32>) -> tensor<512x512xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x330xi8>, tensor<330x512xi8>) outs(%acc: tensor<512x512xi32>) -> tensor<512x512xi32>
  return %result: tensor<512x512xi32>
}

func.func @matmul_accumulate_1x1000xi8_times_1000x1000xi8_into_1x1000xi32(%lhs: tensor<1x1000xi8>, %rhs: tensor<1000x1000xi8>, %acc: tensor<1x1000xi32>) -> tensor<1x1000xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1x1000xi8>, tensor<1000x1000xi8>) outs(%acc: tensor<1x1000xi32>) -> tensor<1x1000xi32>
  return %result: tensor<1x1000xi32>
}

func.func @matmul_accumulate_1000x1000xi8_times_1000x1xi8_into_1000x1xi32(%lhs: tensor<1000x1000xi8>, %rhs: tensor<1000x1xi8>, %acc: tensor<1000x1xi32>) -> tensor<1000x1xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1000x1000xi8>, tensor<1000x1xi8>) outs(%acc: tensor<1000x1xi32>) -> tensor<1000x1xi32>
  return %result: tensor<1000x1xi32>
}

func.func @matmul_1000x1000xi8_times_1000x1xi8_into_1000x1xi32(%lhs: tensor<1000x1000xi8>, %rhs: tensor<1000x1xi8>) -> tensor<1000x1xi32> {
  %init_acc = tensor.empty() : tensor<1000x1xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<1000x1xi32>) -> tensor<1000x1xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1000x1000xi8>, tensor<1000x1xi8>) outs(%acc: tensor<1000x1xi32>) -> tensor<1000x1xi32>
  return %result: tensor<1000x1xi32>
}

