module {
  func.func @test_scatter_elements_with_duplicate_indices(%arg0: !torch.vtensor<[1,5],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ScatterElements"(%arg0, %arg1, %arg2) {torch.onnx.axis = 1 : si64, torch.onnx.reduction = "add"} : (!torch.vtensor<[1,5],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32> 
    return %0 : !torch.vtensor<[1,5],f32>
  }
}

