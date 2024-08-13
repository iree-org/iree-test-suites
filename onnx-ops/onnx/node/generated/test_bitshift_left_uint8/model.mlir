module {
  func.func @test_bitshift_left_uint8(%arg0: !torch.vtensor<[3],ui8>, %arg1: !torch.vtensor<[3],ui8>) -> !torch.vtensor<[3],ui8> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "LEFT"} : (!torch.vtensor<[3],ui8>, !torch.vtensor<[3],ui8>) -> !torch.vtensor<[3],ui8> 
    return %0 : !torch.vtensor<[3],ui8>
  }
}

