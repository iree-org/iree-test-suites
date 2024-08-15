module {
  func.func @test_det_nd(%arg0: !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Det"(%arg0) : (!torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[3],f32> 
    return %0 : !torch.vtensor<[3],f32>
  }
}

