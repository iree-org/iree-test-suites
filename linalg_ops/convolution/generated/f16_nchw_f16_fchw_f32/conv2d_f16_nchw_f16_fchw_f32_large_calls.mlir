builtin.module @calls attributes {
  
} {

func.func private @conv2d_test.generate_random_tensor(%device: !hal.device, %dim0: i64, %dim1: i64, %dim2: i64, %dim3: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view
func.func private @conv2d_test.check_conv2d_results(%device: !hal.device, %n: i64, %c: i64, %h: i64, %w: i64, %f:i64, %kh:i64, %kw:i64, %layout:i64, %sh:i64, %sw:i64, %dh:i64, %dw:i64, %input: !hal.buffer_view, %kernel: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)
func.func private @module.conv2d_accumulate_2_4_128_128_times_3_3_8_dtype_f16_f16_f32(%input: !hal.buffer_view, %kernel: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view
func.func private @module.conv2d_accumulate_2_3_128_128_times_3_3_12_dtype_f16_f16_f32(%input: !hal.buffer_view, %kernel: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view

func.func @conv2d_accumulate_2_4_128_128_times_3_3_8_dtype_f16_f16_f32_2_4_128_128_8_3_3_acc_0() attributes {
  iree.reflection = {description = "Conv2d shape (NxCxHxWxFxKHxKW): 2x4x128x128x8x3x3"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %input_dim0 = arith.constant 2 : i64
  %input_dim1 = arith.constant 4 : i64
  %input_dim2 = arith.constant 128 : i64
  %input_dim3 = arith.constant 128 : i64
  %input_element_type = hal.element_type<f16> : i32
  %input_seed = arith.constant 2 : i32
  %input = call @conv2d_test.generate_random_tensor(%device, %input_dim0, %input_dim1, %input_dim2, %input_dim3, %input_element_type, %input_seed) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view
  %kernel_dim0 = arith.constant 8 : i64
  %kernel_dim1 = arith.constant 4 : i64
  %kernel_dim2 = arith.constant 3 : i64
  %kernel_dim3 = arith.constant 3 : i64
  %kernel_element_type = hal.element_type<f16> : i32
  %kernel_seed = arith.constant 3 : i32
  %kernel = call @conv2d_test.generate_random_tensor(%device, %kernel_dim0, %kernel_dim1, %kernel_dim2, %kernel_dim3, %kernel_element_type, %kernel_seed) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 2 : i64
  %acc_dim1 = arith.constant 8 : i64
  %acc_dim2 = arith.constant 126 : i64
  %acc_dim3 = arith.constant 126 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 4 : i32
  %acc = call @conv2d_test.generate_random_tensor(%device, %acc_dim0, %acc_dim1, %acc_dim2, %acc_dim3, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 2 : i64
  %acc_copy_dim1 = arith.constant 8 : i64
  %acc_copy_dim2 = arith.constant 126 : i64
  %acc_copy_dim3 = arith.constant 126 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 4 : i32
  %acc_copy = call @conv2d_test.generate_random_tensor(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_dim2, %acc_copy_dim3, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.conv2d_accumulate_2_4_128_128_times_3_3_8_dtype_f16_f16_f32(%input, %kernel, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %n = arith.constant 2 : i64
  %c = arith.constant 4 : i64
  %h = arith.constant 128 : i64
  %w = arith.constant 128 : i64
  %f = arith.constant 8 : i64
  %kh = arith.constant 3 : i64
  %kw = arith.constant 3 : i64
  %layout = arith.constant 0 : i64
  %sh = arith.constant 1 : i64
  %sw = arith.constant 1 : i64
  %dh = arith.constant 1 : i64
  %dw = arith.constant 1 : i64
  call @conv2d_test.check_conv2d_results(%device, %n, %c, %h, %w, %f, %kh, %kw, %layout, %sh, %sw, %dh, %dw, %input, %kernel, %acc, %result) : (!hal.device, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}
func.func @conv2d_accumulate_2_3_128_128_times_3_3_12_dtype_f16_f16_f32_2_3_128_128_12_3_3_acc_1() attributes {
  iree.reflection = {description = "Conv2d shape (NxCxHxWxFxKHxKW): 2x3x128x128x12x3x3"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %input_dim0 = arith.constant 2 : i64
  %input_dim1 = arith.constant 3 : i64
  %input_dim2 = arith.constant 128 : i64
  %input_dim3 = arith.constant 128 : i64
  %input_element_type = hal.element_type<f16> : i32
  %input_seed = arith.constant 5 : i32
  %input = call @conv2d_test.generate_random_tensor(%device, %input_dim0, %input_dim1, %input_dim2, %input_dim3, %input_element_type, %input_seed) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view
  %kernel_dim0 = arith.constant 12 : i64
  %kernel_dim1 = arith.constant 3 : i64
  %kernel_dim2 = arith.constant 3 : i64
  %kernel_dim3 = arith.constant 3 : i64
  %kernel_element_type = hal.element_type<f16> : i32
  %kernel_seed = arith.constant 6 : i32
  %kernel = call @conv2d_test.generate_random_tensor(%device, %kernel_dim0, %kernel_dim1, %kernel_dim2, %kernel_dim3, %kernel_element_type, %kernel_seed) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_dim0 = arith.constant 2 : i64
  %acc_dim1 = arith.constant 12 : i64
  %acc_dim2 = arith.constant 126 : i64
  %acc_dim3 = arith.constant 126 : i64
  %acc_element_type = hal.element_type<f32> : i32
  %acc_seed = arith.constant 7 : i32
  %acc = call @conv2d_test.generate_random_tensor(%device, %acc_dim0, %acc_dim1, %acc_dim2, %acc_dim3, %acc_element_type, %acc_seed) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view
  %acc_copy_dim0 = arith.constant 2 : i64
  %acc_copy_dim1 = arith.constant 12 : i64
  %acc_copy_dim2 = arith.constant 126 : i64
  %acc_copy_dim3 = arith.constant 126 : i64
  %acc_copy_element_type = hal.element_type<f32> : i32
  %acc_copy_seed = arith.constant 7 : i32
  %acc_copy = call @conv2d_test.generate_random_tensor(%device, %acc_copy_dim0, %acc_copy_dim1, %acc_copy_dim2, %acc_copy_dim3, %acc_copy_element_type, %acc_copy_seed) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view
  %result = call @module.conv2d_accumulate_2_3_128_128_times_3_3_12_dtype_f16_f16_f32(%input, %kernel, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %n = arith.constant 2 : i64
  %c = arith.constant 3 : i64
  %h = arith.constant 128 : i64
  %w = arith.constant 128 : i64
  %f = arith.constant 12 : i64
  %kh = arith.constant 3 : i64
  %kw = arith.constant 3 : i64
  %layout = arith.constant 0 : i64
  %sh = arith.constant 1 : i64
  %sw = arith.constant 1 : i64
  %dh = arith.constant 1 : i64
  %dw = arith.constant 1 : i64
  call @conv2d_test.check_conv2d_results(%device, %n, %c, %h, %w, %f, %kh, %kw, %layout, %sh, %sw, %dh, %dw, %input, %kernel, %acc, %result) : (!hal.device, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}
}
