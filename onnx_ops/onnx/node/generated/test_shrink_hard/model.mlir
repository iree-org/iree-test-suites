module {
  func.func @test_shrink_hard(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32> attributes {torch.onnx_meta.ir_version = 4 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Shrink"(%arg0) {torch.onnx.lambd = 1.500000e+00 : f32} : (!torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32> 
    return %0 : !torch.vtensor<[5],f32>
  }
}

