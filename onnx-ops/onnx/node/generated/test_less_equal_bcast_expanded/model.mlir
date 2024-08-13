module {
  func.func @test_less_equal_bcast_expanded(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Less"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],i1> 
    %1 = torch.operator "onnx.Equal"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],i1> 
    %2 = torch.operator "onnx.Or"(%0, %1) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[3,4,5],i1>) -> !torch.vtensor<[3,4,5],i1> 
    return %2 : !torch.vtensor<[3,4,5],i1>
  }
}

