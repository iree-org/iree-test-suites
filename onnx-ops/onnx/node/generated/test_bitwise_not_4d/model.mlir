module {
  func.func @test_bitwise_not_4d(%arg0: !torch.vtensor<[3,4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.BitwiseNot"(%arg0) : (!torch.vtensor<[3,4,5,6],ui8>) -> !torch.vtensor<[3,4,5,6],ui8> 
    return %0 : !torch.vtensor<[3,4,5,6],ui8>
  }
}

