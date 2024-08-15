module {
  func.func @test_gru_with_initial_bias(%arg0: !torch.vtensor<[1,3,3],f32>, %arg1: !torch.vtensor<[1,9,3],f32>, %arg2: !torch.vtensor<[1,9,3],f32>, %arg3: !torch.vtensor<[1,18],f32>) -> !torch.vtensor<[1,3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[1,3,3],f32>, !torch.vtensor<[1,9,3],f32>, !torch.vtensor<[1,9,3],f32>, !torch.vtensor<[1,18],f32>) -> (!torch.none, !torch.vtensor<[1,3,3],f32>) 
    return %0#1 : !torch.vtensor<[1,3,3],f32>
  }
}

