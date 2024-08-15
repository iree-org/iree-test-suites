module {
  func.func @test_averagepool_2d_precomputed_pads(%arg0: !torch.vtensor<[1,1,5,5],f32>) -> !torch.vtensor<[1,1,5,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.AveragePool"(%arg0) {torch.onnx.kernel_shape = [5 : si64, 5 : si64], torch.onnx.pads = [2 : si64, 2 : si64, 2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,5,5],f32>) -> !torch.vtensor<[1,1,5,5],f32> 
    return %0 : !torch.vtensor<[1,1,5,5],f32>
  }
}

