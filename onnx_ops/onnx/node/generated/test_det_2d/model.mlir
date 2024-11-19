module {
  func.func @test_det_2d(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Det"(%arg0) : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
}

