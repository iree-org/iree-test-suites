builtin.module @calls attributes {
  
} {

func.func private @matmul_test.generate_random_matrix(%device: !hal.device, %dim0: i64, %dim1: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view
func.func private @matmul_test.check_matmul_results(%device: !hal.device, %m: i64, %k: i64, %n: i64, %transpose_rhs: i32, %lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)

func.func private @module.matmul_accumulate_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_123x456xf16_times_456x789xf16_into_123x789xf16(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_654x321xf16_times_321x234xf16_into_654x234xf16(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_1x1000xf16_times_1000x1000xf16_into_1x1000xf16(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_accumulate_1000x1000xf16_times_1000x1xf16_into_1000x1xf16(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_1000x1000xf16_times_1000x1xf16_into_1000x1xf16(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view

func.func @matmul_accumulate_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16_123_456_789_acc_0() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 123x456x789"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 123 : i64
  %lhs_dim1 = arith.constant 456 : i64
  %lhs_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 2 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 456 : i64
  %rhs_dim1 = arith.constant 789 : i64
  %rhs_element_type = hal.element_type<f16> : i32
  %rhs_seed = arith.constant 3 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 123 : i64
  %acc_dim1 = arith.constant 789 : i64
  %acc_element_type = hal.element_type<f16> : i32
  %acc_seed = arith.constant 4 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 123 : i64
  %acc_copy_dim1 = arith.constant 789 : i64
  %acc_copy_element_type = hal.element_type<f16> : i32
  %acc_copy_seed = arith.constant 4 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 123 : i64
  %k = arith.constant 456 : i64
  %n = arith.constant 789 : i64
  %transpose_rhs = arith.constant 0 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_123x456xf16_times_456x789xf16_into_123x789xf16_123_456_789_acc_1() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 123x456x789"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 123 : i64
  %lhs_dim1 = arith.constant 456 : i64
  %lhs_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 5 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 456 : i64
  %rhs_dim1 = arith.constant 789 : i64
  %rhs_element_type = hal.element_type<f16> : i32
  %rhs_seed = arith.constant 6 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 123 : i64
  %acc_dim1 = arith.constant 789 : i64
  %acc_element_type = hal.element_type<f16> : i32
  %acc_seed = arith.constant 7 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 123 : i64
  %acc_copy_dim1 = arith.constant 789 : i64
  %acc_copy_element_type = hal.element_type<f16> : i32
  %acc_copy_seed = arith.constant 7 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_123x456xf16_times_456x789xf16_into_123x789xf16(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 123 : i64
  %k = arith.constant 456 : i64
  %n = arith.constant 789 : i64
  %transpose_rhs = arith.constant 0 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16_654_321_234_2() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 654x321x234"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 654 : i64
  %lhs_dim1 = arith.constant 321 : i64
  %lhs_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 8 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 321 : i64
  %rhs_dim1 = arith.constant 234 : i64
  %rhs_element_type = hal.element_type<f16> : i32
  %rhs_seed = arith.constant 9 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 654 : i64
  %k = arith.constant 321 : i64
  %n = arith.constant 234 : i64
  %transpose_rhs = arith.constant 0 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_654x321xf16_times_321x234xf16_into_654x234xf16_654_321_234_3() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 654x321x234"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 654 : i64
  %lhs_dim1 = arith.constant 321 : i64
  %lhs_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 10 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 321 : i64
  %rhs_dim1 = arith.constant 234 : i64
  %rhs_element_type = hal.element_type<f16> : i32
  %rhs_seed = arith.constant 11 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_654x321xf16_times_321x234xf16_into_654x234xf16(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 654 : i64
  %k = arith.constant 321 : i64
  %n = arith.constant 234 : i64
  %transpose_rhs = arith.constant 0 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16_1_1000_1000_acc_4() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x1000x1000"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 12 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1000 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<f16> : i32
  %rhs_seed = arith.constant 13 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 1 : i64
  %acc_dim1 = arith.constant 1000 : i64
  %acc_element_type = hal.element_type<f16> : i32
  %acc_seed = arith.constant 14 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 1 : i64
  %acc_copy_dim1 = arith.constant 1000 : i64
  %acc_copy_element_type = hal.element_type<f16> : i32
  %acc_copy_seed = arith.constant 14 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 1000 : i64
  %transpose_rhs = arith.constant 0 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_1x1000xf16_times_1000x1000xf16_into_1x1000xf16_1_1000_1000_acc_5() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1x1000x1000"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 15 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1000 : i64
  %rhs_dim1 = arith.constant 1000 : i64
  %rhs_element_type = hal.element_type<f16> : i32
  %rhs_seed = arith.constant 16 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 1 : i64
  %acc_dim1 = arith.constant 1000 : i64
  %acc_element_type = hal.element_type<f16> : i32
  %acc_seed = arith.constant 17 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 1 : i64
  %acc_copy_dim1 = arith.constant 1000 : i64
  %acc_copy_element_type = hal.element_type<f16> : i32
  %acc_copy_seed = arith.constant 17 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_1x1000xf16_times_1000x1000xf16_into_1x1000xf16(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 1000 : i64
  %transpose_rhs = arith.constant 0 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16_1000_1000_1_acc_6() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1000x1000x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1000 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 18 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1000 : i64
  %rhs_dim1 = arith.constant 1 : i64
  %rhs_element_type = hal.element_type<f16> : i32
  %rhs_seed = arith.constant 19 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 1000 : i64
  %acc_dim1 = arith.constant 1 : i64
  %acc_element_type = hal.element_type<f16> : i32
  %acc_seed = arith.constant 20 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 1000 : i64
  %acc_copy_dim1 = arith.constant 1 : i64
  %acc_copy_element_type = hal.element_type<f16> : i32
  %acc_copy_seed = arith.constant 20 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1000 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 0 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_accumulate_1000x1000xf16_times_1000x1xf16_into_1000x1xf16_1000_1000_1_acc_7() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1000x1000x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1000 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 21 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1000 : i64
  %rhs_dim1 = arith.constant 1 : i64
  %rhs_element_type = hal.element_type<f16> : i32
  %rhs_seed = arith.constant 22 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 1000 : i64
  %acc_dim1 = arith.constant 1 : i64
  %acc_element_type = hal.element_type<f16> : i32
  %acc_seed = arith.constant 23 : i32
  %acc = call @matmul_test.generate_random_matrix(%device, %acc_dim0, %acc_dim1, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 1000 : i64
  %acc_copy_dim1 = arith.constant 1 : i64
  %acc_copy_element_type = hal.element_type<f16> : i32
  %acc_copy_seed = arith.constant 23 : i32
  %acc_copy = call @matmul_test.generate_random_matrix(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.matmul_accumulate_1000x1000xf16_times_1000x1xf16_into_1000x1xf16(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1000 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 0 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16_1000_1000_1_8() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1000x1000x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1000 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 24 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1000 : i64
  %rhs_dim1 = arith.constant 1 : i64
  %rhs_element_type = hal.element_type<f16> : i32
  %rhs_seed = arith.constant 25 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf16(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1000 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 0 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_1000x1000xf16_times_1000x1xf16_into_1000x1xf16_1000_1000_1_9() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 1000x1000x1"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 1000 : i64
  %lhs_dim1 = arith.constant 1000 : i64
  %lhs_element_type = hal.element_type<f16> : i32
  %lhs_seed = arith.constant 26 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 1000 : i64
  %rhs_dim1 = arith.constant 1 : i64
  %rhs_element_type = hal.element_type<f16> : i32
  %rhs_seed = arith.constant 27 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_1000x1000xf16_times_1000x1xf16_into_1000x1xf16(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 1000 : i64
  %k = arith.constant 1000 : i64
  %n = arith.constant 1 : i64
  %transpose_rhs = arith.constant 0 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}


}
