module {
  func.func @test_melweightmatrix(%arg0: !torch.vtensor<[],si32>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[],si32>, %arg3: !torch.vtensor<[],f32>, %arg4: !torch.vtensor<[],f32>) -> !torch.vtensor<[9,8],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.MelWeightMatrix"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.vtensor<[],si32>, !torch.vtensor<[],si32>, !torch.vtensor<[],si32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[9,8],f32> 
    return %0 : !torch.vtensor<[9,8],f32>
  }
}

