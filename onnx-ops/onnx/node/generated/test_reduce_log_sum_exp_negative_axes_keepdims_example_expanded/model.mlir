module {
  func.func @test_reduce_log_sum_exp_negative_axes_keepdims_example_expanded(%arg0: !torch.vtensor<[3,2,2],f64>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 11 : si64} : (!torch.vtensor<[3,2,2],f64>) -> !torch.vtensor<[3,2,2],f64> 
    %1 = torch.operator "onnx.Exp"(%0) : (!torch.vtensor<[3,2,2],f64>) -> !torch.vtensor<[3,2,2],f64> 
    %2 = torch.operator "onnx.ReduceSum"(%1, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[],f64> 
    %3 = torch.operator "onnx.Log"(%2) : (!torch.vtensor<[],f64>) -> !torch.vtensor<[],f64> 
    %4 = torch.operator "onnx.CastLike"(%3, %arg0) : (!torch.vtensor<[],f64>, !torch.vtensor<[3,2,2],f64>) -> !torch.vtensor<[3,1,2],f64> 
    return %4 : !torch.vtensor<[3,1,2],f64>
  }
}

