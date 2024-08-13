module {
  func.func @test_constant_pad_negative_axes(%arg0: !torch.vtensor<[1,3,4,5],f32>, %arg1: !torch.vtensor<[4],si64>, %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,3,4,12],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Pad"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.mode = "constant"} : (!torch.vtensor<[1,3,4,5],f32>, !torch.vtensor<[4],si64>, !torch.vtensor<[],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,3,4,12],f32> 
    return %0 : !torch.vtensor<[1,3,4,12],f32>
  }
}

