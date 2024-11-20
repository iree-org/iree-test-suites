module {
  func.func @test_maxpool_2d_strides(%arg0: !torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,10,10],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.kernel_shape = [5 : si64, 5 : si64], torch.onnx.strides = [3 : si64, 3 : si64]} : (!torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,10,10],f32> 
    return %0 : !torch.vtensor<[1,3,10,10],f32>
  }
}

