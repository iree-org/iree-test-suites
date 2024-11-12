builtin.module @calls attributes {
  
} {

func.func private @attention_test.generate_random_tensor(%device: !hal.device, %dim0: i64, %dim1: i64, %dim2: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view
func.func private @attention_test.check_attention_results(%device: !hal.device, %batch: i64, %m: i64, %k1: i64, %k2: i64, %n: i64, %query: !hal.buffer_view, %key: !hal.buffer_view, %value: !hal.buffer_view, %result: !hal.buffer_view)

func.func private @module.attention_2_256_64_32_16_dtype_f16_f16_f16_f16(%query: !hal.buffer_view, %key: !hal.buffer_view, %value: !hal.buffer_view, %scale: f32) -> !hal.buffer_view

func.func @attention_2_256_64_32_16_dtype_f16_f16_f16_f16_2_256_64_32_16_64_1.0_0() attributes {
  iree.reflection = {description = "Attention shape (BATCHxMxK1xK2xN): 2x256x64x32x64x16"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  %query_dim0 = arith.constant 2 : i64
  %query_dim1 = arith.constant 256 : i64
  %query_dim2 = arith.constant 64 : i64
  %query_element_type = hal.element_type<f16> : i32
  %query_seed = arith.constant 2 : i32
  %query = call @attention_test.generate_random_tensor(%device, %query_dim0, %query_dim1, %query_dim2, %query_element_type, %query_seed) : (!hal.device, i64, i64, i64, i32, i32) -> !hal.buffer_view
  %key_dim0 = arith.constant 2 : i64
  %key_dim1 = arith.constant 32 : i64
  %key_dim2 = arith.constant 64 : i64
  %key_element_type = hal.element_type<f16> : i32
  %key_seed = arith.constant 3 : i32
  %key = call @attention_test.generate_random_tensor(%device, %key_dim0, %key_dim1, %key_dim2, %key_element_type, %key_seed) : (!hal.device, i64, i64, i64, i32, i32) -> !hal.buffer_view
  %value_dim0 = arith.constant 2 : i64
  %value_dim1 = arith.constant 32 : i64
  %value_dim2 = arith.constant 16 : i64
  %value_element_type = hal.element_type<f16> : i32
  %value_seed = arith.constant 4 : i32
  %value = call @attention_test.generate_random_tensor(%device, %value_dim0, %value_dim1, %value_dim2, %value_element_type, %value_seed) : (!hal.device, i64, i64, i64, i32, i32) -> !hal.buffer_view
  %scale = arith.constant 1.0 : f32
  %result = call @module.attention_2_256_64_32_16_dtype_f16_f16_f16_f16(%query, %key, %value, %scale) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, f32) -> !hal.buffer_view
  %batch = arith.constant 2 : i64 
  %m = arith.constant 256 : i64 
  %k1 = arith.constant 64 : i64 
  %k2 = arith.constant 32 : i64 
  %n = arith.constant 16 : i64 
  %queryTensor = hal.tensor.import %query : !hal.buffer_view -> tensor<2x256x64xf16> 
  %keyTensor = hal.tensor.import %key : !hal.buffer_view -> tensor<2x32x64xf16> 
  %valueTensor = hal.tensor.import %value : !hal.buffer_view -> tensor<2x32x16xf16> 
  %resultTensor = hal.tensor.import %result : !hal.buffer_view -> tensor<2x256x16xf16> 
  %queryExt = arith.extf %queryTensor : tensor<2x256x64xf16> to tensor<2x256x64xf32> 
  %keyExt = arith.extf %keyTensor : tensor<2x32x64xf16> to tensor<2x32x64xf32> 
  %valueExt = arith.extf %valueTensor : tensor<2x32x16xf16> to tensor<2x32x16xf32> 
  %resultExt = arith.extf %resultTensor : tensor<2x256x16xf16> to tensor<2x256x16xf32> 
  %queryExtBufferView = hal.tensor.export %queryExt : tensor<2x256x64xf32> -> !hal.buffer_view 
  %keyExtBufferView = hal.tensor.export %keyExt : tensor<2x32x64xf32> -> !hal.buffer_view 
  %valueExtBufferView = hal.tensor.export %valueExt : tensor<2x32x16xf32> -> !hal.buffer_view 
  %resultExtBufferView = hal.tensor.export %resultExt : tensor<2x256x16xf32> -> !hal.buffer_view 
  call @attention_test.check_attention_results(%device, %batch, %m, %k1, %k2, %n, %queryExtBufferView, %keyExtBufferView, %valueExtBufferView, %resultExtBufferView) : (!hal.device, i64, i64, i64, i64, i64, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}
}
