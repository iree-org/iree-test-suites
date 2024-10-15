builtin.module @calls attributes {
  
} {

func.func private @matmul_test.generate_random_matrix(%device: !hal.device, %dim0: i64, %dim1: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view
func.func private @matmul_test.check_matmul_results(%device: !hal.device, %m: i64, %k: i64, %n: i64, %transpose_rhs: i32, %lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)

func.func private @module.matmul_accumulate_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_512x128xbf16_times_512x128xbf16_into_512x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_512x128xbf16_times_512x128xbf16_into_512x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_1000x4xbf16_times_512x4xbf16_into_1000x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_4x1000xbf16_times_512x1000xbf16_into_4x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_512x1000xbf16_times_4x1000xbf16_into_512x4xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_512x128xbf16_times_500x128xbf16_into_512x500xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_457x330xbf16_times_512x330xbf16_into_457x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_457x330xbf16_times_514x330xbf16_into_457x514xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_438x330xbf16_times_514x330xbf16_into_438x514xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_540x332xbf16_times_516x332xbf16_into_540x516xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_654x321xbf16_times_234x321xbf16_into_654x234xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_457x160xbf16_times_512x160xbf16_into_457x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_512x330xbf16_times_512x330xbf16_into_512x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_1x1000xbf16_times_1000x1000xbf16_into_1x1000xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_1000x1000xbf16_times_1x1000xbf16_into_1000x1xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_1000x1000xbf16_times_1x1000xbf16_into_1000x1xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view

func.func @matmul_accumulate_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_512_128_512_acc_0() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x128x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 128 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 2 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 128 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 3 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 512 : i64
  %acc_dim1 = arith.constant 512 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 4 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 512 : i64
  %acc_copy_dim1 = arith.constant 512 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 4 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 512 : i64
  %k = arith.constant 128 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_512x128xbf16_times_512x128xbf16_into_512x512xf32_512_128_512_acc_1() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x128x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 128 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 5 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 128 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 6 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 512 : i64
  %acc_dim1 = arith.constant 512 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 7 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 512 : i64
  %acc_copy_dim1 = arith.constant 512 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 7 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_512x128xbf16_times_512x128xbf16_into_512x512xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 512 : i64
  %k = arith.constant 128 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_512_128_512_2() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x128x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 128 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 8 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 128 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 9 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 512 : i64
  %k = arith.constant 128 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_512x128xbf16_times_512x128xbf16_into_512x512xf32_512_128_512_3() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x128x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 128 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 10 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 128 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 11 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_512x128xbf16_times_512x128xbf16_into_512x512xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 512 : i64
  %k = arith.constant 128 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_1000_4_512_4() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1000x4x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1000 : i64
  %lhs_dim1 = arith.constant 4 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 12 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 4 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 13 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1000 : i64
  %k = arith.constant 4 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_1000x4xbf16_times_512x4xbf16_into_1000x512xf32_1000_4_512_5() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1000x4x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1000 : i64
  %lhs_dim1 = arith.constant 4 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 14 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 4 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 15 : i32
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

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_4_1000_512_6() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 4x1000x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 4 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 16 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 17 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 4 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_4x1000xbf16_times_512x1000xbf16_into_4x512xf32_4_1000_512_7() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 4x1000x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 4 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 18 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 19 : i32
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

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_512_1000_4_8() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x1000x4"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 20 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 4 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 21 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 512 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 4 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_512x1000xbf16_times_4x1000xbf16_into_512x4xf32_512_1000_4_9() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x1000x4"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 22 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 4 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 23 : i32
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

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_512_128_500_10() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x128x500"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 128 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 24 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 500 : i64
  %rhs_dim1 = arith.constant 128 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 25 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 512 : i64
  %k = arith.constant 128 : i64
  %n = arith.constant 500 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_512x128xbf16_times_500x128xbf16_into_512x500xf32_512_128_500_11() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x128x500"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 128 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 26 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 500 : i64
  %rhs_dim1 = arith.constant 128 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 27 : i32
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

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_457_330_512_12() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 457x330x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 457 : i64
  %lhs_dim1 = arith.constant 330 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 28 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 330 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 29 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 457 : i64
  %k = arith.constant 330 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_457x330xbf16_times_512x330xbf16_into_457x512xf32_457_330_512_13() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 457x330x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 457 : i64
  %lhs_dim1 = arith.constant 330 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 30 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 330 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 31 : i32
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

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_457_330_514_14() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 457x330x514"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 457 : i64
  %lhs_dim1 = arith.constant 330 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 32 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 514 : i64
  %rhs_dim1 = arith.constant 330 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 33 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 457 : i64
  %k = arith.constant 330 : i64
  %n = arith.constant 514 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_457x330xbf16_times_514x330xbf16_into_457x514xf32_457_330_514_15() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 457x330x514"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 457 : i64
  %lhs_dim1 = arith.constant 330 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 34 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 514 : i64
  %rhs_dim1 = arith.constant 330 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 35 : i32
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

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_438_330_514_16() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 438x330x514"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 438 : i64
  %lhs_dim1 = arith.constant 330 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 36 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 514 : i64
  %rhs_dim1 = arith.constant 330 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 37 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 438 : i64
  %k = arith.constant 330 : i64
  %n = arith.constant 514 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_438x330xbf16_times_514x330xbf16_into_438x514xf32_438_330_514_17() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 438x330x514"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 438 : i64
  %lhs_dim1 = arith.constant 330 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 38 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 514 : i64
  %rhs_dim1 = arith.constant 330 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 39 : i32
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

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_540_332_516_18() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 540x332x516"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 540 : i64
  %lhs_dim1 = arith.constant 332 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 40 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 516 : i64
  %rhs_dim1 = arith.constant 332 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 41 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 540 : i64
  %k = arith.constant 332 : i64
  %n = arith.constant 516 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_540x332xbf16_times_516x332xbf16_into_540x516xf32_540_332_516_19() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 540x332x516"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 540 : i64
  %lhs_dim1 = arith.constant 332 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 42 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 516 : i64
  %rhs_dim1 = arith.constant 332 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 43 : i32
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

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_654_321_234_20() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 654x321x234"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 654 : i64
  %lhs_dim1 = arith.constant 321 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 44 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 234 : i64
  %rhs_dim1 = arith.constant 321 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 45 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 654 : i64
  %k = arith.constant 321 : i64
  %n = arith.constant 234 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_654x321xbf16_times_234x321xbf16_into_654x234xf32_654_321_234_21() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 654x321x234"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 654 : i64
  %lhs_dim1 = arith.constant 321 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 46 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 234 : i64
  %rhs_dim1 = arith.constant 321 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 47 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_654x321xbf16_times_234x321xbf16_into_654x234xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 654 : i64
  %k = arith.constant 321 : i64
  %n = arith.constant 234 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_457_160_512_22() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 457x160x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 457 : i64
  %lhs_dim1 = arith.constant 160 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 48 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 160 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 49 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 457 : i64
  %k = arith.constant 160 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_457x160xbf16_times_512x160xbf16_into_457x512xf32_457_160_512_23() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 457x160x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 457 : i64
  %lhs_dim1 = arith.constant 160 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 50 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 160 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 51 : i32
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

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_512_330_512_24() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x330x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 330 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 52 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 330 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 53 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 512 : i64
  %k = arith.constant 330 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_512x330xbf16_times_512x330xbf16_into_512x512xf32_512_330_512_25() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x330x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 330 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 54 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 512 : i64
  %rhs_dim1 = arith.constant 330 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 55 : i32
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

func.func @matmul_accumulate_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_1_1000_1000_acc_26() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x1000x1000"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 56 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1000 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 57 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 1 : i64
  %acc_dim1 = arith.constant 1000 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 58 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 1 : i64
  %acc_copy_dim1 = arith.constant 1000 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 58 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 1000 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_1x1000xbf16_times_1000x1000xbf16_into_1x1000xf32_1_1000_1000_acc_27() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x1000x1000"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 59 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1000 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 60 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 1 : i64
  %acc_dim1 = arith.constant 1000 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 61 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 1 : i64
  %acc_copy_dim1 = arith.constant 1000 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 61 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_1x1000xbf16_times_1000x1000xbf16_into_1x1000xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 1000 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_1000_1000_1_acc_28() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1000x1000x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1000 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 62 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 63 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 1000 : i64
  %acc_dim1 = arith.constant 1 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 64 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 1000 : i64
  %acc_copy_dim1 = arith.constant 1 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 64 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1000 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_1000x1000xbf16_times_1x1000xbf16_into_1000x1xf32_1000_1000_1_acc_29() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1000x1000x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1000 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 65 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 66 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 1000 : i64
  %acc_dim1 = arith.constant 1 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 67 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 1000 : i64
  %acc_copy_dim1 = arith.constant 1 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 67 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_1000x1000xbf16_times_1x1000xbf16_into_1000x1xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1000 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32_1000_1000_1_30() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1000x1000x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1000 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 68 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 69 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1000 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_1000x1000xbf16_times_1x1000xbf16_into_1000x1xf32_1000_1000_1_31() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1000x1000x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1000 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<bf16> : i32
  %lhs_seed = arith.constant 70 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<bf16> : i32
  %rhs_seed = arith.constant 71 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_1000x1000xbf16_times_1x1000xbf16_into_1000x1xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1000 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}


}
