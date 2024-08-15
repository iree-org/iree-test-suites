module {
  func.func @test_averagepool_3d_dilations_small(%arg0: !torch.vtensor<[1,1,4,4,4],f32>) -> !torch.vtensor<[1,1,2,2,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.AveragePool"(%arg0) {torch.onnx.ceil_mode = 1 : si64, torch.onnx.dilations = [2 : si64, 2 : si64, 2 : si64], torch.onnx.kernel_shape = [2 : si64, 2 : si64, 2 : si64], torch.onnx.strides = [1 : si64, 1 : si64, 1 : si64]} : (!torch.vtensor<[1,1,4,4,4],f32>) -> !torch.vtensor<[1,1,2,2,2],f32> 
    return %0 : !torch.vtensor<[1,1,2,2,2],f32>
  }
}

