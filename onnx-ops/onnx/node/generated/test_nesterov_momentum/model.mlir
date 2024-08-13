module {
  func.func @test_nesterov_momentum(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],si64>, %arg2: !torch.vtensor<[2],f32>, %arg3: !torch.vtensor<[2],f32>, %arg4: !torch.vtensor<[2],f32>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_versions = {ai.onnx.preview.training = 1 : si64}, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:2 = torch.operator "onnx.Momentum"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.alpha = 0.949999988 : f32, torch.onnx.beta = 1.000000e+00 : f32, torch.onnx.mode = "nesterov", torch.onnx.norm_coefficient = 0.00999999977 : f32} : (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) 
    return %0#0, %0#1 : !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>
  }
}

