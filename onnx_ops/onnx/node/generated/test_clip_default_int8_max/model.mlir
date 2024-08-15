module {
  func.func @test_clip_default_int8_max(%arg0: !torch.vtensor<[3,4,5],si8>, %arg1: !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Clip"(%arg0, %none, %arg1) : (!torch.vtensor<[3,4,5],si8>, !torch.none, !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> 
    return %0 : !torch.vtensor<[3,4,5],si8>
  }
}

