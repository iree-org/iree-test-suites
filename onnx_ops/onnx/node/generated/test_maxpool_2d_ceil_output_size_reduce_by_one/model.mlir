module {
  func.func @test_maxpool_2d_ceil_output_size_reduce_by_one(%arg0: !torch.vtensor<[1,1,2,2],f32>) -> !torch.vtensor<[1,1,1,1],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.ceil_mode = 1 : si64, torch.onnx.kernel_shape = [1 : si64, 1 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,2,2],f32>) -> !torch.vtensor<[1,1,1,1],f32> 
    return %0 : !torch.vtensor<[1,1,1,1],f32>
  }
}

