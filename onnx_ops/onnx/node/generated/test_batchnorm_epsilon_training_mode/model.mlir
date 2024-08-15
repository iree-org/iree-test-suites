module {
  func.func @test_batchnorm_epsilon_training_mode(%arg0: !torch.vtensor<[2,3,4,5],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>, %arg3: !torch.vtensor<[3],f32>, %arg4: !torch.vtensor<[3],f32>) -> (!torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:3 = torch.operator "onnx.BatchNormalization"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.epsilon = 0.00999999977 : f32, torch.onnx.training_mode = 1 : si64} : (!torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> (!torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) 
    return %0#0, %0#1, %0#2 : !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>
  }
}

