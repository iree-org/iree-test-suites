module {
  func.func @test_globalaveragepool_precomputed(%arg0: !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,1,1],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.GlobalAveragePool"(%arg0) : (!torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,1,1],f32> 
    return %0 : !torch.vtensor<[1,1,1,1],f32>
  }
}

