module {
  func.func @test_argmax_no_keepdims_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.axis = 1 : si64, torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64> 
    return %0 : !torch.vtensor<[2],si64>
  }
}

