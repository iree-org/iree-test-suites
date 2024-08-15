module {
  func.func @test_bitwise_and_i16_3d(%arg0: !torch.vtensor<[3,4,5],si16>, %arg1: !torch.vtensor<[3,4,5],si16>) -> !torch.vtensor<[3,4,5],si16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.BitwiseAnd"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],si16>, !torch.vtensor<[3,4,5],si16>) -> !torch.vtensor<[3,4,5],si16> 
    return %0 : !torch.vtensor<[3,4,5],si16>
  }
}

