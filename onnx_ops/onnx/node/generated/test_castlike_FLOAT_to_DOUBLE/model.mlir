module {
  func.func @test_castlike_FLOAT_to_DOUBLE(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[1],f64>) -> !torch.vtensor<[3,4],f64> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.CastLike"(%arg0, %arg1) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],f64>) -> !torch.vtensor<[3,4],f64> 
    return %0 : !torch.vtensor<[3,4],f64>
  }
}

