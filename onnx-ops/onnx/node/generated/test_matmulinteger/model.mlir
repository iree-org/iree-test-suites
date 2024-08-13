module {
  func.func @test_matmulinteger(%arg0: !torch.vtensor<[4,3],ui8>, %arg1: !torch.vtensor<[3,2],ui8>, %arg2: !torch.vtensor<[1],ui8>, %arg3: !torch.vtensor<[1],ui8>) -> !torch.vtensor<[4,2],si32> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (!torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,2],ui8>, !torch.vtensor<[1],ui8>, !torch.vtensor<[1],ui8>) -> !torch.vtensor<[4,2],si32> 
    return %0 : !torch.vtensor<[4,2],si32>
  }
}

