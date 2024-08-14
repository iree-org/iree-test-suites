module {
  func.func @test_squeeze(%arg0: !torch.vtensor<[1,3,4,5],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Squeeze"(%arg0, %arg1) : (!torch.vtensor<[1,3,4,5],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,4,5],f32> 
    return %0 : !torch.vtensor<[3,4,5],f32>
  }
}

