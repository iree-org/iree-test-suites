module {
  func.func @test_tile(%arg0: !torch.vtensor<[2,3,4,5],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[14,18,16,10],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Tile"(%arg0, %arg1) : (!torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[14,18,16,10],f32> 
    return %0 : !torch.vtensor<[14,18,16,10],f32>
  }
}

