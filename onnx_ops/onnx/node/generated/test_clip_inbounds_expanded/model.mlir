module {
  func.func @test_clip_inbounds_expanded(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Less"(%arg0, %arg1) : (!torch.vtensor<[3],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3],i1> 
    %1 = torch.operator "onnx.Where"(%0, %arg1, %arg0) : (!torch.vtensor<[3],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> 
    %2 = torch.operator "onnx.Less"(%arg2, %1) : (!torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],i1> 
    %3 = torch.operator "onnx.Where"(%2, %arg2, %1) : (!torch.vtensor<[3],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> 
    return %3 : !torch.vtensor<[3],f32>
  }
}

