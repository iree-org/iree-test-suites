module {
  func.func @test_hannwindow_symmetric_expanded(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<5.000000e-01> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<5.000000e-01> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %6 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<6.28318548> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %7 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[],si32>) -> !torch.vtensor<[],f32> 
    %8 = torch.operator "onnx.Sub"(%7, %4) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %9 = torch.operator "onnx.Constant"() {torch.onnx.value_int = 0 : si64} : () -> !torch.vtensor<[],si64> 
    %10 = torch.operator "onnx.Cast"(%9) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> 
    %11 = torch.operator "onnx.Sub"(%4, %10) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %12 = torch.operator "onnx.Mul"(%7, %10) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %13 = torch.operator "onnx.Mul"(%8, %11) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %14 = torch.operator "onnx.Add"(%12, %13) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %15 = torch.operator "onnx.Div"(%6, %14) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %16 = torch.operator "onnx.Range"(%3, %7, %4) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[?],f32> 
    %17 = torch.operator "onnx.Mul"(%16, %15) : (!torch.vtensor<[?],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[?],f32> 
    %18 = torch.operator "onnx.Mul"(%17, %5) : (!torch.vtensor<[?],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[?],f32> 
    %19 = torch.operator "onnx.Cos"(%18) : (!torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> 
    %20 = torch.operator "onnx.Mul"(%2, %19) : (!torch.vtensor<[],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> 
    %21 = torch.operator "onnx.Cos"(%17) : (!torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> 
    %22 = torch.operator "onnx.Mul"(%1, %21) : (!torch.vtensor<[],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> 
    %23 = torch.operator "onnx.Sub"(%0, %22) : (!torch.vtensor<[],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> 
    %24 = torch.operator "onnx.Add"(%23, %20) : (!torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> 
    %25 = torch.operator "onnx.Cast"(%24) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[?],f32>) -> !torch.vtensor<[10],f32> 
    return %25 : !torch.vtensor<[10],f32>
  }
}

