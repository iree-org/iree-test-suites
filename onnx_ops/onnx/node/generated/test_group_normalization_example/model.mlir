module {
  func.func @test_group_normalization_example(%arg0: !torch.vtensor<[3,4,2,2],f32>, %arg1: !torch.vtensor<[4],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[3,4,2,2],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.GroupNormalization"(%arg0, %arg1, %arg2) {torch.onnx.num_groups = 2 : si64} : (!torch.vtensor<[3,4,2,2],f32>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[3,4,2,2],f32> 
    return %0 : !torch.vtensor<[3,4,2,2],f32>
  }
}

