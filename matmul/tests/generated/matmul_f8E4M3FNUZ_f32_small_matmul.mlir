func.func @matmul_accumulate_DYNxDYNxf8E4M3FNUZ_times_DYNxDYNxf8E4M3FNUZ_into_DYNxDYNxf32(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<?x?xf32> to tensor<?x?xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<?x?xf32> to tensor<?x?xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<?x?xf8E4M3FNUZ>, tensor<?x?xf8E4M3FNUZ>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>  return %result: tensor<?x?xf32>
}

func.func @matmul_accumulate_1x1xf8E4M3FNUZ_times_1x1xf8E4M3FNUZ_into_1x1xf32(%lhs: tensor<1x1xf32>, %rhs: tensor<1x1xf32>, %acc: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<1x1xf32> to tensor<1x1xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<1x1xf32> to tensor<1x1xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<1x1xf8E4M3FNUZ>, tensor<1x1xf8E4M3FNUZ>) outs(%acc: tensor<1x1xf32>) -> tensor<1x1xf32>  return %result: tensor<1x1xf32>
}

func.func @matmul_DYNxDYNxf8E4M3FNUZ_times_DYNxDYNxf8E4M3FNUZ_into_DYNxDYNxf32(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc_dim0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %acc_dim1 = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : tensor<?x?xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %lhs_casted = arith.truncf %lhs: tensor<?x?xf32> to tensor<?x?xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<?x?xf32> to tensor<?x?xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<?x?xf8E4M3FNUZ>, tensor<?x?xf8E4M3FNUZ>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>  return %result: tensor<?x?xf32>
}

func.func @matmul_1x1xf8E4M3FNUZ_times_1x1xf8E4M3FNUZ_into_1x1xf32(%lhs: tensor<1x1xf32>, %rhs: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %init_acc = tensor.empty() : tensor<1x1xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<1x1xf32>) -> tensor<1x1xf32>
  %lhs_casted = arith.truncf %lhs: tensor<1x1xf32> to tensor<1x1xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<1x1xf32> to tensor<1x1xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<1x1xf8E4M3FNUZ>, tensor<1x1xf8E4M3FNUZ>) outs(%acc: tensor<1x1xf32>) -> tensor<1x1xf32>  return %result: tensor<1x1xf32>
}

func.func @matmul_accumulate_2x2xf8E4M3FNUZ_times_2x2xf8E4M3FNUZ_into_2x2xf32(%lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>, %acc: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<2x2xf32> to tensor<2x2xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<2x2xf32> to tensor<2x2xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<2x2xf8E4M3FNUZ>, tensor<2x2xf8E4M3FNUZ>) outs(%acc: tensor<2x2xf32>) -> tensor<2x2xf32>  return %result: tensor<2x2xf32>
}

func.func @matmul_accumulate_4x4xf8E4M3FNUZ_times_4x4xf8E4M3FNUZ_into_4x4xf32(%lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>, %acc: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<4x4xf32> to tensor<4x4xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<4x4xf32> to tensor<4x4xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<4x4xf8E4M3FNUZ>, tensor<4x4xf8E4M3FNUZ>) outs(%acc: tensor<4x4xf32>) -> tensor<4x4xf32>  return %result: tensor<4x4xf32>
}

func.func @matmul_accumulate_8x8xf8E4M3FNUZ_times_8x8xf8E4M3FNUZ_into_8x8xf32(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<8x8xf32> to tensor<8x8xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<8x8xf32> to tensor<8x8xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<8x8xf8E4M3FNUZ>, tensor<8x8xf8E4M3FNUZ>) outs(%acc: tensor<8x8xf32>) -> tensor<8x8xf32>  return %result: tensor<8x8xf32>
}

func.func @matmul_accumulate_9x9xf8E4M3FNUZ_times_9x9xf8E4M3FNUZ_into_9x9xf32(%lhs: tensor<9x9xf32>, %rhs: tensor<9x9xf32>, %acc: tensor<9x9xf32>) -> tensor<9x9xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<9x9xf32> to tensor<9x9xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<9x9xf32> to tensor<9x9xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<9x9xf8E4M3FNUZ>, tensor<9x9xf8E4M3FNUZ>) outs(%acc: tensor<9x9xf32>) -> tensor<9x9xf32>  return %result: tensor<9x9xf32>
}

func.func @matmul_accumulate_6x13xf8E4M3FNUZ_times_13x3xf8E4M3FNUZ_into_6x3xf32(%lhs: tensor<6x13xf32>, %rhs: tensor<13x3xf32>, %acc: tensor<6x3xf32>) -> tensor<6x3xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<6x13xf32> to tensor<6x13xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<13x3xf32> to tensor<13x3xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<6x13xf8E4M3FNUZ>, tensor<13x3xf8E4M3FNUZ>) outs(%acc: tensor<6x3xf32>) -> tensor<6x3xf32>  return %result: tensor<6x3xf32>
}

func.func @matmul_15x37xf8E4M3FNUZ_times_37x7xf8E4M3FNUZ_into_15x7xf32(%lhs: tensor<15x37xf32>, %rhs: tensor<37x7xf32>) -> tensor<15x7xf32> {
  %init_acc = tensor.empty() : tensor<15x7xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<15x7xf32>) -> tensor<15x7xf32>
  %lhs_casted = arith.truncf %lhs: tensor<15x37xf32> to tensor<15x37xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<37x7xf32> to tensor<37x7xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<15x37xf8E4M3FNUZ>, tensor<37x7xf8E4M3FNUZ>) outs(%acc: tensor<15x7xf32>) -> tensor<15x7xf32>  return %result: tensor<15x7xf32>
}

func.func @matmul_accumulate_81x19xf8E4M3FNUZ_times_19x41xf8E4M3FNUZ_into_81x41xf32(%lhs: tensor<81x19xf32>, %rhs: tensor<19x41xf32>, %acc: tensor<81x41xf32>) -> tensor<81x41xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<81x19xf32> to tensor<81x19xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<19x41xf32> to tensor<19x41xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<81x19xf8E4M3FNUZ>, tensor<19x41xf8E4M3FNUZ>) outs(%acc: tensor<81x41xf32>) -> tensor<81x41xf32>  return %result: tensor<81x41xf32>
}

func.func @matmul_accumulate_1x10xf8E4M3FNUZ_times_10x10xf8E4M3FNUZ_into_1x10xf32(%lhs: tensor<1x10xf32>, %rhs: tensor<10x10xf32>, %acc: tensor<1x10xf32>) -> tensor<1x10xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<1x10xf32> to tensor<1x10xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<10x10xf32> to tensor<10x10xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<1x10xf8E4M3FNUZ>, tensor<10x10xf8E4M3FNUZ>) outs(%acc: tensor<1x10xf32>) -> tensor<1x10xf32>  return %result: tensor<1x10xf32>
}

func.func @matmul_1x10xf8E4M3FNUZ_times_10x10xf8E4M3FNUZ_into_1x10xf32(%lhs: tensor<1x10xf32>, %rhs: tensor<10x10xf32>) -> tensor<1x10xf32> {
  %init_acc = tensor.empty() : tensor<1x10xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<1x10xf32>) -> tensor<1x10xf32>
  %lhs_casted = arith.truncf %lhs: tensor<1x10xf32> to tensor<1x10xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<10x10xf32> to tensor<10x10xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<1x10xf8E4M3FNUZ>, tensor<10x10xf8E4M3FNUZ>) outs(%acc: tensor<1x10xf32>) -> tensor<1x10xf32>  return %result: tensor<1x10xf32>
}

func.func @matmul_accumulate_10x1xf8E4M3FNUZ_times_1x10xf8E4M3FNUZ_into_10x10xf32(%lhs: tensor<10x1xf32>, %rhs: tensor<1x10xf32>, %acc: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<10x1xf32> to tensor<10x1xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<1x10xf32> to tensor<1x10xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<10x1xf8E4M3FNUZ>, tensor<1x10xf8E4M3FNUZ>) outs(%acc: tensor<10x10xf32>) -> tensor<10x10xf32>  return %result: tensor<10x10xf32>
}

func.func @matmul_accumulate_10x10xf8E4M3FNUZ_times_10x1xf8E4M3FNUZ_into_10x1xf32(%lhs: tensor<10x10xf32>, %rhs: tensor<10x1xf32>, %acc: tensor<10x1xf32>) -> tensor<10x1xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<10x10xf32> to tensor<10x10xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<10x1xf32> to tensor<10x1xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<10x10xf8E4M3FNUZ>, tensor<10x1xf8E4M3FNUZ>) outs(%acc: tensor<10x1xf32>) -> tensor<10x1xf32>  return %result: tensor<10x1xf32>
}

func.func @matmul_10x10xf8E4M3FNUZ_times_10x1xf8E4M3FNUZ_into_10x1xf32(%lhs: tensor<10x10xf32>, %rhs: tensor<10x1xf32>) -> tensor<10x1xf32> {
  %init_acc = tensor.empty() : tensor<10x1xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<10x1xf32>) -> tensor<10x1xf32>
  %lhs_casted = arith.truncf %lhs: tensor<10x10xf32> to tensor<10x10xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<10x1xf32> to tensor<10x1xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<10x10xf8E4M3FNUZ>, tensor<10x1xf8E4M3FNUZ>) outs(%acc: tensor<10x1xf32>) -> tensor<10x1xf32>  return %result: tensor<10x1xf32>
}

