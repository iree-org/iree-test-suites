module {
  func.func @test_reduce_l2_do_not_keepdims_example_expanded(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Mul"(%arg0, %arg0) : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[3,2,2],f32> 
    %1 = torch.operator "onnx.ReduceSum"(%0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Cast"(%1) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %3 = torch.operator "onnx.Sqrt"(%2) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %4 = torch.operator "onnx.CastLike"(%3, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[3,2],f32> 
    return %4 : !torch.vtensor<[3,2],f32>
  }
}

