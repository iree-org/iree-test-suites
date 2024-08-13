module {
  func.func @test_where_long_example(%arg0: !torch.vtensor<[2,2],i1>, %arg1: !torch.vtensor<[2,2],si64>, %arg2: !torch.vtensor<[2,2],si64>) -> !torch.vtensor<[2,2],si64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Where"(%arg0, %arg1, %arg2) : (!torch.vtensor<[2,2],i1>, !torch.vtensor<[2,2],si64>, !torch.vtensor<[2,2],si64>) -> !torch.vtensor<[2,2],si64> 
    return %0 : !torch.vtensor<[2,2],si64>
  }
}

