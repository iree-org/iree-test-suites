module {
  func.func @test_reduce_min_default_axes_keepdims_random(%arg0: !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ReduceMin"(%arg0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[1,1,1],f32> 
    return %0 : !torch.vtensor<[1,1,1],f32>
  }
}

