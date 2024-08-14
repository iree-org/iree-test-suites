module {
  func.func @test_pow_types_float32_uint32(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],ui32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Pow"(%arg0, %arg1) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],ui32>) -> !torch.vtensor<[3],f32> 
    return %0 : !torch.vtensor<[3],f32>
  }
}

