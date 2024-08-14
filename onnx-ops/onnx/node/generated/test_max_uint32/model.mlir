module {
  func.func @test_max_uint32(%arg0: !torch.vtensor<[3],ui32>, %arg1: !torch.vtensor<[3],ui32>) -> !torch.vtensor<[3],ui32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Max"(%arg0, %arg1) : (!torch.vtensor<[3],ui32>, !torch.vtensor<[3],ui32>) -> !torch.vtensor<[3],ui32> 
    return %0 : !torch.vtensor<[3],ui32>
  }
}

