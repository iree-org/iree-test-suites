module {
  func.func @test_argmax_default_axis_random(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[1,3,4],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[1,3,4],si64> 
    return %0 : !torch.vtensor<[1,3,4],si64>
  }
}

