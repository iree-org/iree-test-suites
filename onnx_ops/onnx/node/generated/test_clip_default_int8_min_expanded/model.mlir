module {
  func.func @test_clip_default_int8_min_expanded(%arg0: !torch.vtensor<[3,4,5],si8>, %arg1: !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Less"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si8>, !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],i1> 
    %1 = torch.operator "onnx.Where"(%0, %arg1, %arg0) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[],si8>, !torch.vtensor<[3,4,5],si8>) -> !torch.vtensor<[3,4,5],si8> 
    return %1 : !torch.vtensor<[3,4,5],si8>
  }
}

