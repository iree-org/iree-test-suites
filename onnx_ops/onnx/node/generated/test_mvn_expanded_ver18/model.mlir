module {
  func.func @test_mvn_expanded_ver18(%arg0: !torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[3,3,3,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<9.99999971E-10> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value_ints = [0 : si64, 2 : si64, 3 : si64]} : () -> !torch.vtensor<[3],si64> 
    %3 = torch.operator "onnx.ReduceMean"(%arg0, %2) : (!torch.vtensor<[3,3,3,1],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,3,1,1],f32> 
    %4 = torch.operator "onnx.Pow"(%3, %0) : (!torch.vtensor<[1,3,1,1],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,1,1],f32> 
    %5 = torch.operator "onnx.Pow"(%arg0, %0) : (!torch.vtensor<[3,3,3,1],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3,3,3,1],f32> 
    %6 = torch.operator "onnx.ReduceMean"(%5, %2) : (!torch.vtensor<[3,3,3,1],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,3,1,1],f32> 
    %7 = torch.operator "onnx.Sub"(%6, %4) : (!torch.vtensor<[1,3,1,1],f32>, !torch.vtensor<[1,3,1,1],f32>) -> !torch.vtensor<[1,3,1,1],f32> 
    %8 = torch.operator "onnx.Sqrt"(%7) : (!torch.vtensor<[1,3,1,1],f32>) -> !torch.vtensor<[1,3,1,1],f32> 
    %9 = torch.operator "onnx.Sub"(%arg0, %3) : (!torch.vtensor<[3,3,3,1],f32>, !torch.vtensor<[1,3,1,1],f32>) -> !torch.vtensor<[3,3,3,1],f32> 
    %10 = torch.operator "onnx.Add"(%8, %1) : (!torch.vtensor<[1,3,1,1],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,1,1],f32> 
    %11 = torch.operator "onnx.Div"(%9, %10) : (!torch.vtensor<[3,3,3,1],f32>, !torch.vtensor<[1,3,1,1],f32>) -> !torch.vtensor<[3,3,3,1],f32> 
    return %11 : !torch.vtensor<[3,3,3,1],f32>
  }
}

