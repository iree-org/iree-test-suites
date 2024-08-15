module {
  func.func @test_sce_mean_no_weight_ii_4d_log_prob(%arg0: !torch.vtensor<[3,5,2,7],f32>, %arg1: !torch.vtensor<[3,2,7],si64>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[3,5,2,7],f32>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:2 = torch.operator "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1) {torch.onnx.ignore_index = 2 : si64, torch.onnx.reduction = "mean"} : (!torch.vtensor<[3,5,2,7],f32>, !torch.vtensor<[3,2,7],si64>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[3,5,2,7],f32>) 
    return %0#0, %0#1 : !torch.vtensor<[],f32>, !torch.vtensor<[3,5,2,7],f32>
  }
}

