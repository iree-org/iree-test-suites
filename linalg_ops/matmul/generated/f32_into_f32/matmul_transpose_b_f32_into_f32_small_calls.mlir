builtin.module @calls attributes {
  
} {

func.func private @matmul_test.generate_random_matrix(%device: !hal.device, %dim0: i64, %dim1: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view
func.func private @matmul_test.check_matmul_results(%device: !hal.device, %m: i64, %k: i64, %n: i64, %transpose_rhs: i32, %lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)

func.func private @module.matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_1x1xf32_times_1x1xf32_into_1x1xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_1x1xf32_times_1x1xf32_into_1x1xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_2x2xf32_times_2x2xf32_into_2x2xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_4x4xf32_times_4x4xf32_into_4x4xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_8x8xf32_times_8x8xf32_into_8x8xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_9x9xf32_times_9x9xf32_into_9x9xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_6x13xf32_times_3x13xf32_into_6x3xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_15x37xf32_times_7x37xf32_into_15x7xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_81x19xf32_times_41x19xf32_into_81x41xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_1x10xf32_times_10x10xf32_into_1x10xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_1x10xf32_times_10x10xf32_into_1x10xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_10x1xf32_times_10x1xf32_into_10x10xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_10x10xf32_times_1x10xf32_into_10x1xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_10x10xf32_times_1x10xf32_into_10x1xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view

func.func @matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_1_1_1_acc_0() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x1x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1 : i64
  %lhs_dim1 = arith.constant 1 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 2 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1 : i64
  %rhs_dim1 = arith.constant 1 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 3 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 1 : i64
  %acc_dim1 = arith.constant 1 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 4 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 1 : i64
  %acc_copy_dim1 = arith.constant 1 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 4 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 1 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_1x1xf32_times_1x1xf32_into_1x1xf32_1_1_1_acc_1() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x1x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1 : i64
  %lhs_dim1 = arith.constant 1 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 5 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1 : i64
  %rhs_dim1 = arith.constant 1 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 6 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 1 : i64
  %acc_dim1 = arith.constant 1 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 7 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 1 : i64
  %acc_copy_dim1 = arith.constant 1 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 7 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_1x1xf32_times_1x1xf32_into_1x1xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 1 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_1_1_1_2() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x1x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1 : i64
  %lhs_dim1 = arith.constant 1 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 8 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1 : i64
  %rhs_dim1 = arith.constant 1 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 9 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 1 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_1x1xf32_times_1x1xf32_into_1x1xf32_1_1_1_3() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x1x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1 : i64
  %lhs_dim1 = arith.constant 1 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 10 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1 : i64
  %rhs_dim1 = arith.constant 1 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 11 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_1x1xf32_times_1x1xf32_into_1x1xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 1 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_2_2_2_acc_4() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 2x2x2"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 2 : i64
  %lhs_dim1 = arith.constant 2 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 12 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 2 : i64
  %rhs_dim1 = arith.constant 2 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 13 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 2 : i64
  %acc_dim1 = arith.constant 2 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 14 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 2 : i64
  %acc_copy_dim1 = arith.constant 2 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 14 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 2 : i64
  %k = arith.constant 2 : i64
  %n = arith.constant 2 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_2x2xf32_times_2x2xf32_into_2x2xf32_2_2_2_acc_5() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 2x2x2"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 2 : i64
  %lhs_dim1 = arith.constant 2 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 15 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 2 : i64
  %rhs_dim1 = arith.constant 2 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 16 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 2 : i64
  %acc_dim1 = arith.constant 2 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 17 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 2 : i64
  %acc_copy_dim1 = arith.constant 2 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 17 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_2x2xf32_times_2x2xf32_into_2x2xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 2 : i64
  %k = arith.constant 2 : i64
  %n = arith.constant 2 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_4_4_4_acc_6() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 4x4x4"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 4 : i64
  %lhs_dim1 = arith.constant 4 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 18 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 4 : i64
  %rhs_dim1 = arith.constant 4 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 19 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 4 : i64
  %acc_dim1 = arith.constant 4 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 20 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 4 : i64
  %acc_copy_dim1 = arith.constant 4 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 20 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 4 : i64
  %k = arith.constant 4 : i64
  %n = arith.constant 4 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_4x4xf32_times_4x4xf32_into_4x4xf32_4_4_4_acc_7() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 4x4x4"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 4 : i64
  %lhs_dim1 = arith.constant 4 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 21 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 4 : i64
  %rhs_dim1 = arith.constant 4 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 22 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 4 : i64
  %acc_dim1 = arith.constant 4 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 23 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 4 : i64
  %acc_copy_dim1 = arith.constant 4 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 23 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_4x4xf32_times_4x4xf32_into_4x4xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 4 : i64
  %k = arith.constant 4 : i64
  %n = arith.constant 4 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_8_8_8_acc_8() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 8x8x8"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 8 : i64
  %lhs_dim1 = arith.constant 8 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 24 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 8 : i64
  %rhs_dim1 = arith.constant 8 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 25 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 8 : i64
  %acc_dim1 = arith.constant 8 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 26 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 8 : i64
  %acc_copy_dim1 = arith.constant 8 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 26 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 8 : i64
  %k = arith.constant 8 : i64
  %n = arith.constant 8 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_8x8xf32_times_8x8xf32_into_8x8xf32_8_8_8_acc_9() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 8x8x8"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 8 : i64
  %lhs_dim1 = arith.constant 8 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 27 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 8 : i64
  %rhs_dim1 = arith.constant 8 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 28 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 8 : i64
  %acc_dim1 = arith.constant 8 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 29 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 8 : i64
  %acc_copy_dim1 = arith.constant 8 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 29 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_8x8xf32_times_8x8xf32_into_8x8xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 8 : i64
  %k = arith.constant 8 : i64
  %n = arith.constant 8 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_9_9_9_acc_10() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 9x9x9"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 9 : i64
  %lhs_dim1 = arith.constant 9 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 30 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 9 : i64
  %rhs_dim1 = arith.constant 9 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 31 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 9 : i64
  %acc_dim1 = arith.constant 9 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 32 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 9 : i64
  %acc_copy_dim1 = arith.constant 9 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 32 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 9 : i64
  %k = arith.constant 9 : i64
  %n = arith.constant 9 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_9x9xf32_times_9x9xf32_into_9x9xf32_9_9_9_acc_11() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 9x9x9"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 9 : i64
  %lhs_dim1 = arith.constant 9 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 33 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 9 : i64
  %rhs_dim1 = arith.constant 9 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 34 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 9 : i64
  %acc_dim1 = arith.constant 9 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 35 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 9 : i64
  %acc_copy_dim1 = arith.constant 9 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 35 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_9x9xf32_times_9x9xf32_into_9x9xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 9 : i64
  %k = arith.constant 9 : i64
  %n = arith.constant 9 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_6_13_3_acc_12() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 6x13x3"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 6 : i64
  %lhs_dim1 = arith.constant 13 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 36 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 3 : i64
  %rhs_dim1 = arith.constant 13 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 37 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 6 : i64
  %acc_dim1 = arith.constant 3 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 38 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 6 : i64
  %acc_copy_dim1 = arith.constant 3 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 38 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 6 : i64
  %k = arith.constant 13 : i64
  %n = arith.constant 3 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_6x13xf32_times_3x13xf32_into_6x3xf32_6_13_3_acc_13() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 6x13x3"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 6 : i64
  %lhs_dim1 = arith.constant 13 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 39 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 3 : i64
  %rhs_dim1 = arith.constant 13 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 40 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 6 : i64
  %acc_dim1 = arith.constant 3 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 41 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 6 : i64
  %acc_copy_dim1 = arith.constant 3 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 41 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_6x13xf32_times_3x13xf32_into_6x3xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 6 : i64
  %k = arith.constant 13 : i64
  %n = arith.constant 3 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_15_37_7_14() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 15x37x7"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 15 : i64
  %lhs_dim1 = arith.constant 37 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 42 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 7 : i64
  %rhs_dim1 = arith.constant 37 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 43 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 15 : i64
  %k = arith.constant 37 : i64
  %n = arith.constant 7 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_15x37xf32_times_7x37xf32_into_15x7xf32_15_37_7_15() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 15x37x7"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 15 : i64
  %lhs_dim1 = arith.constant 37 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 44 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 7 : i64
  %rhs_dim1 = arith.constant 37 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 45 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_15x37xf32_times_7x37xf32_into_15x7xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 15 : i64
  %k = arith.constant 37 : i64
  %n = arith.constant 7 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_81_19_41_acc_16() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 81x19x41"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 81 : i64
  %lhs_dim1 = arith.constant 19 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 46 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 41 : i64
  %rhs_dim1 = arith.constant 19 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 47 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 81 : i64
  %acc_dim1 = arith.constant 41 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 48 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 81 : i64
  %acc_copy_dim1 = arith.constant 41 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 48 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 81 : i64
  %k = arith.constant 19 : i64
  %n = arith.constant 41 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_81x19xf32_times_41x19xf32_into_81x41xf32_81_19_41_acc_17() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 81x19x41"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 81 : i64
  %lhs_dim1 = arith.constant 19 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 49 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 41 : i64
  %rhs_dim1 = arith.constant 19 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 50 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 81 : i64
  %acc_dim1 = arith.constant 41 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 51 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 81 : i64
  %acc_copy_dim1 = arith.constant 41 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 51 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_81x19xf32_times_41x19xf32_into_81x41xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 81 : i64
  %k = arith.constant 19 : i64
  %n = arith.constant 41 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_1_10_10_acc_18() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x10x10"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1 : i64
  %lhs_dim1 = arith.constant 10 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 52 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 10 : i64
  %rhs_dim1 = arith.constant 10 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 53 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 1 : i64
  %acc_dim1 = arith.constant 10 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 54 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 1 : i64
  %acc_copy_dim1 = arith.constant 10 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 54 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 10 : i64
  %n = arith.constant 10 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_1x10xf32_times_10x10xf32_into_1x10xf32_1_10_10_acc_19() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x10x10"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1 : i64
  %lhs_dim1 = arith.constant 10 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 55 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 10 : i64
  %rhs_dim1 = arith.constant 10 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 56 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 1 : i64
  %acc_dim1 = arith.constant 10 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 57 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 1 : i64
  %acc_copy_dim1 = arith.constant 10 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 57 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_1x10xf32_times_10x10xf32_into_1x10xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 10 : i64
  %n = arith.constant 10 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_1_10_10_20() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x10x10"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1 : i64
  %lhs_dim1 = arith.constant 10 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 58 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 10 : i64
  %rhs_dim1 = arith.constant 10 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 59 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 10 : i64
  %n = arith.constant 10 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_1x10xf32_times_10x10xf32_into_1x10xf32_1_10_10_21() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x10x10"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1 : i64
  %lhs_dim1 = arith.constant 10 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 60 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 10 : i64
  %rhs_dim1 = arith.constant 10 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 61 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_1x10xf32_times_10x10xf32_into_1x10xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 10 : i64
  %n = arith.constant 10 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_10_1_10_acc_22() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 10x1x10"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 10 : i64
  %lhs_dim1 = arith.constant 1 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 62 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 10 : i64
  %rhs_dim1 = arith.constant 1 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 63 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 10 : i64
  %acc_dim1 = arith.constant 10 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 64 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 10 : i64
  %acc_copy_dim1 = arith.constant 10 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 64 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 10 : i64
  %k = arith.constant 1 : i64
  %n = arith.constant 10 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_10x1xf32_times_10x1xf32_into_10x10xf32_10_1_10_acc_23() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 10x1x10"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 10 : i64
  %lhs_dim1 = arith.constant 1 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 65 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 10 : i64
  %rhs_dim1 = arith.constant 1 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 66 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 10 : i64
  %acc_dim1 = arith.constant 10 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 67 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 10 : i64
  %acc_copy_dim1 = arith.constant 10 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 67 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_10x1xf32_times_10x1xf32_into_10x10xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 10 : i64
  %k = arith.constant 1 : i64
  %n = arith.constant 10 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_10_10_1_acc_24() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 10x10x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 10 : i64
  %lhs_dim1 = arith.constant 10 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 68 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1 : i64
  %rhs_dim1 = arith.constant 10 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 69 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 10 : i64
  %acc_dim1 = arith.constant 1 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 70 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 10 : i64
  %acc_copy_dim1 = arith.constant 1 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 70 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 10 : i64
  %k = arith.constant 10 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_10x10xf32_times_1x10xf32_into_10x1xf32_10_10_1_acc_25() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 10x10x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 10 : i64
  %lhs_dim1 = arith.constant 10 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 71 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1 : i64
  %rhs_dim1 = arith.constant 10 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 72 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 10 : i64
  %acc_dim1 = arith.constant 1 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 73 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 10 : i64
  %acc_copy_dim1 = arith.constant 1 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 73 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_10x10xf32_times_1x10xf32_into_10x1xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 10 : i64
  %k = arith.constant 10 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32_10_10_1_26() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 10x10x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 10 : i64
  %lhs_dim1 = arith.constant 10 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 74 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1 : i64
  %rhs_dim1 = arith.constant 10 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 75 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 10 : i64
  %k = arith.constant 10 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_10x10xf32_times_1x10xf32_into_10x1xf32_10_10_1_27() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 10x10x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 10 : i64
  %lhs_dim1 = arith.constant 10 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 76 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1 : i64
  %rhs_dim1 = arith.constant 10 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 77 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_10x10xf32_times_1x10xf32_into_10x1xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 10 : i64
  %k = arith.constant 10 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}


}
