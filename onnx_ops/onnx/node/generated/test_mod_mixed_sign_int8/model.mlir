module {
  func.func @test_mod_mixed_sign_int8(%arg0: !torch.vtensor<[6],si8>, %arg1: !torch.vtensor<[6],si8>) -> !torch.vtensor<[6],si8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Mod"(%arg0, %arg1) : (!torch.vtensor<[6],si8>, !torch.vtensor<[6],si8>) -> !torch.vtensor<[6],si8> 
    return %0 : !torch.vtensor<[6],si8>
  }
}

