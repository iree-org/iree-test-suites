func.func @matmul_accumulate_DYNxDYNxi8_times_DYNxDYNxi8_into_DYNxDYNxi32(%lhs: tensor<?x?xi8>, %rhs: tensor<?x?xi8>, %acc: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<?x?xi8>, tensor<?x?xi8>) outs(%acc: tensor<?x?xi32>) -> tensor<?x?xi32>

  return %result: tensor<?x?xi32>
}

func.func @matmul_accumulate_1x1xi8_times_1x1xi8_into_1x1xi32(%lhs: tensor<1x1xi8>, %rhs: tensor<1x1xi8>, %acc: tensor<1x1xi32>) -> tensor<1x1xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1x1xi8>, tensor<1x1xi8>) outs(%acc: tensor<1x1xi32>) -> tensor<1x1xi32>

  return %result: tensor<1x1xi32>
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

func.func @matmul_1x1xi8_times_1x1xi8_into_1x1xi32(%lhs: tensor<1x1xi8>, %rhs: tensor<1x1xi8>) -> tensor<1x1xi32> {
  %init_acc = tensor.empty() : tensor<1x1xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<1x1xi32>) -> tensor<1x1xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1x1xi8>, tensor<1x1xi8>) outs(%acc: tensor<1x1xi32>) -> tensor<1x1xi32>
  return %result: tensor<1x1xi32>
}

func.func @matmul_accumulate_2x2xi8_times_2x2xi8_into_2x2xi32(%lhs: tensor<2x2xi8>, %rhs: tensor<2x2xi8>, %acc: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<2x2xi8>, tensor<2x2xi8>) outs(%acc: tensor<2x2xi32>) -> tensor<2x2xi32>

  return %result: tensor<2x2xi32>
}

func.func @matmul_accumulate_4x4xi8_times_4x4xi8_into_4x4xi32(%lhs: tensor<4x4xi8>, %rhs: tensor<4x4xi8>, %acc: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<4x4xi8>, tensor<4x4xi8>) outs(%acc: tensor<4x4xi32>) -> tensor<4x4xi32>

  return %result: tensor<4x4xi32>
}

func.func @matmul_accumulate_8x8xi8_times_8x8xi8_into_8x8xi32(%lhs: tensor<8x8xi8>, %rhs: tensor<8x8xi8>, %acc: tensor<8x8xi32>) -> tensor<8x8xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<8x8xi8>, tensor<8x8xi8>) outs(%acc: tensor<8x8xi32>) -> tensor<8x8xi32>

  return %result: tensor<8x8xi32>
}

func.func @matmul_accumulate_9x9xi8_times_9x9xi8_into_9x9xi32(%lhs: tensor<9x9xi8>, %rhs: tensor<9x9xi8>, %acc: tensor<9x9xi32>) -> tensor<9x9xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<9x9xi8>, tensor<9x9xi8>) outs(%acc: tensor<9x9xi32>) -> tensor<9x9xi32>

  return %result: tensor<9x9xi32>
}

func.func @matmul_accumulate_6x13xi8_times_13x3xi8_into_6x3xi32(%lhs: tensor<6x13xi8>, %rhs: tensor<13x3xi8>, %acc: tensor<6x3xi32>) -> tensor<6x3xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<6x13xi8>, tensor<13x3xi8>) outs(%acc: tensor<6x3xi32>) -> tensor<6x3xi32>

  return %result: tensor<6x3xi32>
}

func.func @matmul_15x37xi8_times_37x7xi8_into_15x7xi32(%lhs: tensor<15x37xi8>, %rhs: tensor<37x7xi8>) -> tensor<15x7xi32> {
  %init_acc = tensor.empty() : tensor<15x7xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<15x7xi32>) -> tensor<15x7xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<15x37xi8>, tensor<37x7xi8>) outs(%acc: tensor<15x7xi32>) -> tensor<15x7xi32>
  return %result: tensor<15x7xi32>
}

func.func @matmul_accumulate_81x19xi8_times_19x41xi8_into_81x41xi32(%lhs: tensor<81x19xi8>, %rhs: tensor<19x41xi8>, %acc: tensor<81x41xi32>) -> tensor<81x41xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<81x19xi8>, tensor<19x41xi8>) outs(%acc: tensor<81x41xi32>) -> tensor<81x41xi32>

  return %result: tensor<81x41xi32>
}

func.func @matmul_accumulate_1x10xi8_times_10x10xi8_into_1x10xi32(%lhs: tensor<1x10xi8>, %rhs: tensor<10x10xi8>, %acc: tensor<1x10xi32>) -> tensor<1x10xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1x10xi8>, tensor<10x10xi8>) outs(%acc: tensor<1x10xi32>) -> tensor<1x10xi32>

  return %result: tensor<1x10xi32>
}

func.func @matmul_1x10xi8_times_10x10xi8_into_1x10xi32(%lhs: tensor<1x10xi8>, %rhs: tensor<10x10xi8>) -> tensor<1x10xi32> {
  %init_acc = tensor.empty() : tensor<1x10xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<1x10xi32>) -> tensor<1x10xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1x10xi8>, tensor<10x10xi8>) outs(%acc: tensor<1x10xi32>) -> tensor<1x10xi32>
  return %result: tensor<1x10xi32>
}

func.func @matmul_accumulate_10x1xi8_times_1x10xi8_into_10x10xi32(%lhs: tensor<10x1xi8>, %rhs: tensor<1x10xi8>, %acc: tensor<10x10xi32>) -> tensor<10x10xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<10x1xi8>, tensor<1x10xi8>) outs(%acc: tensor<10x10xi32>) -> tensor<10x10xi32>

  return %result: tensor<10x10xi32>
}

func.func @matmul_accumulate_10x10xi8_times_10x1xi8_into_10x1xi32(%lhs: tensor<10x10xi8>, %rhs: tensor<10x1xi8>, %acc: tensor<10x1xi32>) -> tensor<10x1xi32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<10x10xi8>, tensor<10x1xi8>) outs(%acc: tensor<10x1xi32>) -> tensor<10x1xi32>

  return %result: tensor<10x1xi32>
}

func.func @matmul_10x10xi8_times_10x1xi8_into_10x1xi32(%lhs: tensor<10x10xi8>, %rhs: tensor<10x1xi8>) -> tensor<10x1xi32> {
  %init_acc = tensor.empty() : tensor<10x1xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<10x1xi32>) -> tensor<10x1xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<10x10xi8>, tensor<10x1xi8>) outs(%acc: tensor<10x1xi32>) -> tensor<10x1xi32>
  return %result: tensor<10x1xi32>
}

