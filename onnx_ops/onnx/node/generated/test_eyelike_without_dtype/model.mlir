module {
  func.func @test_eyelike_without_dtype(%arg0: !torch.vtensor<[4,4],si32>) -> !torch.vtensor<[4,4],si32> attributes {torch.onnx_meta.ir_version = 4 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.EyeLike"(%arg0) : (!torch.vtensor<[4,4],si32>) -> !torch.vtensor<[4,4],si32> 
    return %0 : !torch.vtensor<[4,4],si32>
  }
}

