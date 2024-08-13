module {
  func.func @test_nonzero_example(%arg0: !torch.vtensor<[2,2],i1>) -> !torch.vtensor<[2,3],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.NonZero"(%arg0) : (!torch.vtensor<[2,2],i1>) -> !torch.vtensor<[2,3],si64> 
    return %0 : !torch.vtensor<[2,3],si64>
  }
}

