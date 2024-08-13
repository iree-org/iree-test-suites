module {
  func.func @test_shrink_soft_expanded_ver18(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value_float = 1.500000e+00 : f32} : () -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.CastLike"(%0, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value_float = 1.500000e+00 : f32} : () -> !torch.vtensor<[],f32> 
    %3 = torch.operator "onnx.CastLike"(%2, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %5 = torch.operator "onnx.CastLike"(%4, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> 
    %6 = torch.operator "onnx.Neg"(%1) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %7 = torch.operator "onnx.Less"(%arg0, %6) : (!torch.vtensor<[5],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[5],i1> 
    %8 = torch.operator "onnx.Add"(%arg0, %3) : (!torch.vtensor<[5],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[5],f32> 
    %9 = torch.operator "onnx.Sub"(%arg0, %3) : (!torch.vtensor<[5],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[5],f32> 
    %10 = torch.operator "onnx.Less"(%1, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[5],i1> 
    %11 = torch.operator "onnx.Where"(%10, %9, %5) : (!torch.vtensor<[5],i1>, !torch.vtensor<[5],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[5],f32> 
    %12 = torch.operator "onnx.Where"(%7, %8, %11) : (!torch.vtensor<[5],i1>, !torch.vtensor<[5],f32>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32> 
    return %12 : !torch.vtensor<[5],f32>
  }
}

