module {
  func.func @test_layer_normalization_3d_axis_negative_2_epsilon(%arg0: !torch.vtensor<[2,3,5],f32>, %arg1: !torch.vtensor<[3,5],f32>, %arg2: !torch.vtensor<[3,5],f32>) -> (!torch.vtensor<[2,3,5],f32>, !torch.vtensor<[2,1,1],f32>, !torch.vtensor<[2,1,1],f32>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:3 = torch.operator "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {torch.onnx.axis = -2 : si64, torch.onnx.epsilon = 1.000000e-01 : f32} : (!torch.vtensor<[2,3,5],f32>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[3,5],f32>) -> (!torch.vtensor<[2,3,5],f32>, !torch.vtensor<[2,1,1],f32>, !torch.vtensor<[2,1,1],f32>) 
    return %0#0, %0#1, %0#2 : !torch.vtensor<[2,3,5],f32>, !torch.vtensor<[2,1,1],f32>, !torch.vtensor<[2,1,1],f32>
  }
}

