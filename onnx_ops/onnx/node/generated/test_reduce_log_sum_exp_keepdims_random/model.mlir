module {
  func.func @test_reduce_log_sum_exp_keepdims_random(%arg0: !torch.vtensor<[3,2,2],f64>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ReduceLogSumExp"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f64> 
    return %0 : !torch.vtensor<[3,1,2],f64>
  }
}

