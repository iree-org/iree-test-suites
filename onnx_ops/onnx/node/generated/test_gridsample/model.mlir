module {
  func.func @test_gridsample(%arg0: !torch.vtensor<[1,1,4,4],f32>, %arg1: !torch.vtensor<[1,6,6,2],f32>) -> !torch.vtensor<[1,1,6,6],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.GridSample"(%arg0, %arg1) {torch.onnx.align_corners = 0 : si64, torch.onnx.mode = "linear", torch.onnx.padding_mode = "zeros"} : (!torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,6,6,2],f32>) -> !torch.vtensor<[1,1,6,6],f32> 
    return %0 : !torch.vtensor<[1,1,6,6],f32>
  }
}

