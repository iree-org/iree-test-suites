module {
  func.func @test_adam(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],si64>, %arg2: !torch.vtensor<[2],f32>, %arg3: !torch.vtensor<[2],f32>, %arg4: !torch.vtensor<[2],f32>, %arg5: !torch.vtensor<[2],f32>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_versions = {ai.onnx.preview.training = 1 : si64}, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:3 = torch.operator "onnx.Adam"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {torch.onnx.alpha = 0.949999988 : f32, torch.onnx.beta = 1.000000e-01 : f32, torch.onnx.epsilon = 1.000000e-07 : f32, torch.onnx.norm_coefficient = 1.000000e-03 : f32} : (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) 
    return %0#0, %0#1, %0#2 : !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>
  }
}

