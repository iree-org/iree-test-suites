module {
  func.func @test_range_int32_type_negative_delta(%arg0: !torch.vtensor<[],si32>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[],si32>) -> !torch.vtensor<[2],si32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Range"(%arg0, %arg1, %arg2) : (!torch.vtensor<[],si32>, !torch.vtensor<[],si32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[2],si32> 
    return %0 : !torch.vtensor<[2],si32>
  }
}

