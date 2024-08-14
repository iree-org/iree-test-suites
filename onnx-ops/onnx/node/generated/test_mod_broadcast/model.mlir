module {
  func.func @test_mod_broadcast(%arg0: !torch.vtensor<[3,2,5],si32>, %arg1: !torch.vtensor<[1],si32>) -> !torch.vtensor<[3,2,5],si32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Mod"(%arg0, %arg1) : (!torch.vtensor<[3,2,5],si32>, !torch.vtensor<[1],si32>) -> !torch.vtensor<[3,2,5],si32> 
    return %0 : !torch.vtensor<[3,2,5],si32>
  }
}

