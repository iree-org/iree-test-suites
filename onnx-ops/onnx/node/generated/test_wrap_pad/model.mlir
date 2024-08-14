module {
  func.func @test_wrap_pad(%arg0: !torch.vtensor<[1,3,4,5],si32>, %arg1: !torch.vtensor<[8],si64>) -> !torch.vtensor<[1,3,6,7],si32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Pad"(%arg0, %arg1) {torch.onnx.mode = "wrap"} : (!torch.vtensor<[1,3,4,5],si32>, !torch.vtensor<[8],si64>) -> !torch.vtensor<[1,3,6,7],si32> 
    return %0 : !torch.vtensor<[1,3,6,7],si32>
  }
}

