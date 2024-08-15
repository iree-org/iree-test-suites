module {
  func.func @test_maxpool_3d_dilations_use_ref_impl_large(%arg0: !torch.vtensor<[1,1,32,32,32],f32>) -> !torch.vtensor<[1,1,9,9,9],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.ceil_mode = 1 : si64, torch.onnx.dilations = [2 : si64, 2 : si64, 2 : si64], torch.onnx.kernel_shape = [5 : si64, 5 : si64, 5 : si64], torch.onnx.strides = [3 : si64, 3 : si64, 3 : si64]} : (!torch.vtensor<[1,1,32,32,32],f32>) -> !torch.vtensor<[1,1,9,9,9],f32> 
    return %0 : !torch.vtensor<[1,1,9,9,9],f32>
  }
}

