builtin.module @calls attributes {
  
} {

func.func private @matmul_test.generate_random_matrix(%device: !hal.device, %dim0: i64, %dim1: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view
func.func private @matmul_test.check_matmul_results(%device: !hal.device, %m: i64, %k: i64, %n: i64, %transpose_rhs: i32, %lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)

func.func private @module.matmul_457x330xbf16_times_512x330xbf16_into_457x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_457x330xbf16_times_514x330xbf16_into_457x514xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_438x330xbf16_times_514x330xbf16_into_438x514xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_540x332xbf16_times_516x332xbf16_into_540x516xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_1000x4xbf16_times_512x4xbf16_into_1000x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_4x1000xbf16_times_512x1000xbf16_into_4x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_512x1000xbf16_times_4x1000xbf16_into_512x4xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_512x128xbf16_times_500x128xbf16_into_512x500xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_457x160xbf16_times_512x160xbf16_into_457x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_512x330xbf16_times_512x330xbf16_into_512x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view

func.func @matmul_457x330xbf16_times_512x330xbf16_into_457x512xf32_457_330_512_0() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 457x330x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 457 : i64
  %lhs_dim1 = arith.constant 330 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 2 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 330 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 3 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_457x330xbf16_times_512x330xbf16_into_457x512xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 457 : i64
  %k = arith.constant 330 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_457x330xbf16_times_514x330xbf16_into_457x514xf32_457_330_514_1() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 457x330x514"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 457 : i64
  %lhs_dim1 = arith.constant 330 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 4 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 514 : i64
  %rhs_dim1 = arith.constant 330 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 5 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_457x330xbf16_times_514x330xbf16_into_457x514xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 457 : i64
  %k = arith.constant 330 : i64
  %n = arith.constant 514 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_438x330xbf16_times_514x330xbf16_into_438x514xf32_438_330_514_2() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 438x330x514"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 438 : i64
  %lhs_dim1 = arith.constant 330 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 6 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 514 : i64
  %rhs_dim1 = arith.constant 330 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 7 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_438x330xbf16_times_514x330xbf16_into_438x514xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 438 : i64
  %k = arith.constant 330 : i64
  %n = arith.constant 514 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_540x332xbf16_times_516x332xbf16_into_540x516xf32_540_332_516_3() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 540x332x516"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 540 : i64
  %lhs_dim1 = arith.constant 332 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 8 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 516 : i64
  %rhs_dim1 = arith.constant 332 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 9 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_540x332xbf16_times_516x332xbf16_into_540x516xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 540 : i64
  %k = arith.constant 332 : i64
  %n = arith.constant 516 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_1000x4xbf16_times_512x4xbf16_into_1000x512xf32_1000_4_512_4() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1000x4x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1000 : i64
  %lhs_dim1 = arith.constant 4 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 10 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 4 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 11 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_1000x4xbf16_times_512x4xbf16_into_1000x512xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1000 : i64
  %k = arith.constant 4 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_4x1000xbf16_times_512x1000xbf16_into_4x512xf32_4_1000_512_5() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 4x1000x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 4 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 12 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 13 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_4x1000xbf16_times_512x1000xbf16_into_4x512xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 4 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_512x1000xbf16_times_4x1000xbf16_into_512x4xf32_512_1000_4_6() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x1000x4"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 14 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 4 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 15 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_512x1000xbf16_times_4x1000xbf16_into_512x4xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 512 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 4 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_512x128xbf16_times_500x128xbf16_into_512x500xf32_512_128_500_7() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x128x500"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 128 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 16 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 500 : i64
  %rhs_dim1 = arith.constant 128 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 17 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_512x128xbf16_times_500x128xbf16_into_512x500xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 512 : i64
  %k = arith.constant 128 : i64
  %n = arith.constant 500 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_457x160xbf16_times_512x160xbf16_into_457x512xf32_457_160_512_8() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 457x160x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 457 : i64
  %lhs_dim1 = arith.constant 160 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 18 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 160 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 19 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_457x160xbf16_times_512x160xbf16_into_457x512xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 457 : i64
  %k = arith.constant 160 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_512x330xbf16_times_512x330xbf16_into_512x512xf32_512_330_512_9() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x330x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 330 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 20 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 330 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 21 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_512x330xbf16_times_512x330xbf16_into_512x512xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 512 : i64
  %k = arith.constant 330 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}


}
