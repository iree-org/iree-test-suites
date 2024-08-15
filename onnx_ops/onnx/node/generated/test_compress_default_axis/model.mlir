module {
  func.func @test_compress_default_axis(%arg0: !torch.vtensor<[3,2],f32>, %arg1: !torch.vtensor<[5],i1>) -> !torch.vtensor<[2],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Compress"(%arg0, %arg1) : (!torch.vtensor<[3,2],f32>, !torch.vtensor<[5],i1>) -> !torch.vtensor<[2],f32> 
    return %0 : !torch.vtensor<[2],f32>
  }
}

