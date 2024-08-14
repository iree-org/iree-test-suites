module {
  func.func @test_range_float_type_positive_delta_expanded(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[2],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Sub"(%arg1, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.Cast"(%0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Cast"(%arg2) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %3 = torch.operator "onnx.Div"(%1, %2) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %4 = torch.operator "onnx.Ceil"(%3) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %5 = torch.operator "onnx.Relu"(%4) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %6 = torch.operator "onnx.Cast"(%5) {torch.onnx.to = 7 : si64} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],si64> 
    %7 = torch.operator "onnx.Cast"(%5) {torch.onnx.to = 9 : si64} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],i1> 
    %8:2 = torch.operator "onnx.Loop"(%6, %7, %arg0) : (!torch.vtensor<[],si64>, !torch.vtensor<[],i1>, !torch.vtensor<[],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[2],f32>) {
    ^bb0(%arg3: !torch.vtensor<[],si64>, %arg4: !torch.vtensor<[],i1>, %arg5: !torch.vtensor<[],f32>):
      %9 = torch.operator "onnx.Identity"(%arg4) : (!torch.vtensor<[],i1>) -> !torch.vtensor<[],i1> 
      %10 = torch.operator "onnx.Add"(%arg5, %arg2) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
      %11 = torch.operator "onnx.Identity"(%arg5) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
      torch.operator_terminator %9, %10, %11 : !torch.vtensor<[],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
    }
    return %8#1 : !torch.vtensor<[2],f32>
  }
}

