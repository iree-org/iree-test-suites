module {
  func.func @test_layer_normalization_2d_axis_negative_1_expanded(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[4],f32>, %arg2: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,1],f32>, !torch.vtensor<[3,1],f32>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<9.99999974E-6> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.Cast"(%0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Shape"(%arg0) : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2],si64> 
    %3 = torch.operator "onnx.Size"(%2) : (!torch.vtensor<[2],si64>) -> !torch.vtensor<[],si64> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<-1> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %6 = torch.operator "onnx.Slice"(%2, %4, %5) : (!torch.vtensor<[2],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> 
    %7 = torch.operator "onnx.Neg"(%5) : (!torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> 
    %8 = torch.operator "onnx.ConstantOfShape"(%7) {torch.onnx.value = dense<1> : tensor<1xsi64>} : (!torch.vtensor<[1],si64>) -> !torch.vtensor<[?],si64> 
    %9 = torch.operator "onnx.Concat"(%6, %8) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[1],si64>, !torch.vtensor<[?],si64>) -> !torch.vtensor<[?],si64> 
    %10 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> 
    %11 = torch.operator "onnx.Cast"(%10) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> 
    %12 = torch.operator "onnx.ReduceMean"(%11) {torch.onnx.axes = [1 : si64]} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,1],f32> 
    %13 = torch.operator "onnx.Mul"(%11, %11) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> 
    %14 = torch.operator "onnx.ReduceMean"(%13) {torch.onnx.axes = [1 : si64]} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,1],f32> 
    %15 = torch.operator "onnx.Mul"(%12, %12) : (!torch.vtensor<[3,1],f32>, !torch.vtensor<[3,1],f32>) -> !torch.vtensor<[3,1],f32> 
    %16 = torch.operator "onnx.Sub"(%14, %15) : (!torch.vtensor<[3,1],f32>, !torch.vtensor<[3,1],f32>) -> !torch.vtensor<[3,1],f32> 
    %17 = torch.operator "onnx.Add"(%16, %1) : (!torch.vtensor<[3,1],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3,1],f32> 
    %18 = torch.operator "onnx.Sqrt"(%17) : (!torch.vtensor<[3,1],f32>) -> !torch.vtensor<[3,1],f32> 
    %19 = torch.operator "onnx.Sub"(%11, %12) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,1],f32>) -> !torch.vtensor<[3,4],f32> 
    %20 = torch.operator "onnx.Div"(%19, %18) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,1],f32>) -> !torch.vtensor<[3,4],f32> 
    %21 = torch.operator "onnx.Cast"(%20) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> 
    %22 = torch.operator "onnx.Flatten"(%arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4],f32> 
    %23 = torch.operator "onnx.Mul"(%21, %22) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> 
    %24 = torch.operator "onnx.Flatten"(%arg2) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4],f32> 
    %25 = torch.operator "onnx.Add"(%23, %24) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> 
    %26 = torch.operator "onnx.Reshape"(%25, %2) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,4],f32> 
    %27 = torch.operator "onnx.Reciprocal"(%18) : (!torch.vtensor<[3,1],f32>) -> !torch.vtensor<[3,1],f32> 
    %28 = torch.operator "onnx.Reshape"(%12, %9) : (!torch.vtensor<[3,1],f32>, !torch.vtensor<[?],si64>) -> !torch.vtensor<[3,1],f32> 
    %29 = torch.operator "onnx.Reshape"(%27, %9) : (!torch.vtensor<[3,1],f32>, !torch.vtensor<[?],si64>) -> !torch.vtensor<[3,1],f32> 
    return %26, %28, %29 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,1],f32>, !torch.vtensor<[3,1],f32>
  }
}

