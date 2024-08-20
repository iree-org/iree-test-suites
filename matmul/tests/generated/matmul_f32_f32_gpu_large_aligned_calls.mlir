builtin.module @calls attributes {
  
} {

func.func private @matmul_test.generate_random_matrix(%device: !hal.device, %dim0: i64, %dim1: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view
func.func private @matmul_test.check_matmul_results(%device: !hal.device, %m: i64, %k: i64, %n: i64, %transpose_rhs: i32, %lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)

func.func private @module.matmul_accumulate_512x128xf32_times_128x512xf32_into_512x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.matmul_512x128xf32_times_128x512xf32_into_512x512xf32(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view

func.func @matmul_accumulate_512x128xf32_times_128x512xf32_into_512x512xf32_512_128_512_acc_0() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x128x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 128 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 2 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 128 : i64
  %rhs_dim1 = arith.constant 512 : i64
  %rhs_element_type = hal.element_type<f32> : i32
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
  %result = call @module.matmul_accumulate_512x128xf32_times_128x512xf32_into_512x512xf32(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 512 : i64
  %k = arith.constant 128 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 0 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

func.func @matmul_512x128xf32_times_128x512xf32_into_512x512xf32_512_128_512_1() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 512x128x512"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %lhs_dim0 = arith.constant 512 : i64
  %lhs_dim1 = arith.constant 128 : i64
  %lhs_element_type = hal.element_type<f32> : i32
  %lhs_seed = arith.constant 5 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 128 : i64
  %rhs_dim1 = arith.constant 512 : i64
  %rhs_element_type = hal.element_type<f32> : i32
  %rhs_seed = arith.constant 6 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul_512x128xf32_times_128x512xf32_into_512x512xf32(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 512 : i64
  %k = arith.constant 128 : i64
  %n = arith.constant 512 : i64
  %transpose_rhs = arith.constant 0 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}


}
