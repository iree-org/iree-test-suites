module {
  func.func @test_isinf_negative(%arg0: !torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.IsInf"(%arg0) {torch.onnx.detect_positive = 0 : si64} : (!torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1> 
    return %0 : !torch.vtensor<[6],i1>
  }
}

