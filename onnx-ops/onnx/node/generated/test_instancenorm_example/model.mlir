module {
  func.func @test_instancenorm_example(%arg0: !torch.vtensor<[1,2,1,3],f32>, %arg1: !torch.vtensor<[2],f32>, %arg2: !torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2,1,3],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) : (!torch.vtensor<[1,2,1,3],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2,1,3],f32> 
    return %0 : !torch.vtensor<[1,2,1,3],f32>
  }
}

