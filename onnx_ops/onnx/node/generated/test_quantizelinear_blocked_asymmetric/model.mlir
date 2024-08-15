module {
  func.func @test_quantizelinear_blocked_asymmetric(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[3,2],f32>, %arg2: !torch.vtensor<[3,2],ui8>) -> !torch.vtensor<[3,4],ui8> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) {torch.onnx.axis = 1 : si64, torch.onnx.block_size = 2 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,2],f32>, !torch.vtensor<[3,2],ui8>) -> !torch.vtensor<[3,4],ui8> 
    return %0 : !torch.vtensor<[3,4],ui8>
  }
}

