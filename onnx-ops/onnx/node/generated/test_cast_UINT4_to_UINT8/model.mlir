module {
  func.func @test_cast_UINT4_to_UINT8(%arg0: !torch.vtensor<[5,5],ui4>) -> !torch.vtensor<[5,5],ui8> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 2 : si64} : (!torch.vtensor<[5,5],ui4>) -> !torch.vtensor<[5,5],ui8> 
    return %0 : !torch.vtensor<[5,5],ui8>
  }
}

