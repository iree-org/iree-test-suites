module {
  func.func @test_group_normalization_example_expanded(%arg0: !torch.vtensor<[3,4,2,2],f32>, %arg1: !torch.vtensor<[4],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[3,4,2,2],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<9.99999974E-6> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32> 
    %1 = torch.operator "onnx.Cast"(%0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> 
    %2 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[3,4,2,2],f32>) -> !torch.vtensor<[3,4,2,2],f32> 
    %3 = torch.operator "onnx.Shape"(%2) : (!torch.vtensor<[3,4,2,2],f32>) -> !torch.vtensor<[4],si64> 
    %4 = torch.operator "onnx.Shape"(%arg0) {torch.onnx.end = 2 : si64, torch.onnx.start = 1 : si64} : (!torch.vtensor<[3,4,2,2],f32>) -> !torch.vtensor<[1],si64> 
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %6 = torch.operator "onnx.Div"(%4, %5) : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> 
    %7 = torch.operator "onnx.Shape"(%arg0) {torch.onnx.end = 1 : si64, torch.onnx.start = 0 : si64} : (!torch.vtensor<[3,4,2,2],f32>) -> !torch.vtensor<[1],si64> 
    %8 = torch.operator "onnx.Shape"(%arg0) {torch.onnx.start = 2 : si64} : (!torch.vtensor<[3,4,2,2],f32>) -> !torch.vtensor<[2],si64> 
    %9 = torch.operator "onnx.Concat"(%7, %5, %6, %8) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[5],si64> 
    %10 = torch.operator "onnx.Reshape"(%2, %9) : (!torch.vtensor<[3,4,2,2],f32>, !torch.vtensor<[5],si64>) -> !torch.vtensor<[?,?,?,?,?],f32> 
    %11 = torch.operator "onnx.Constant"() {torch.onnx.value_ints = [0 : si64, 0 : si64, -1 : si64]} : () -> !torch.vtensor<[3],si64> 
    %12 = torch.operator "onnx.Reshape"(%10, %11) : (!torch.vtensor<[?,?,?,?,?],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[?,?,?],f32> 
    %13 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %14 = torch.operator "onnx.ReduceMean"(%12, %13) : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[?,?,1],f32> 
    %15 = torch.operator "onnx.Mul"(%12, %12) : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> 
    %16 = torch.operator "onnx.ReduceMean"(%15, %13) : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[?,?,1],f32> 
    %17 = torch.operator "onnx.Mul"(%14, %14) : (!torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>) -> !torch.vtensor<[?,?,1],f32> 
    %18 = torch.operator "onnx.Sub"(%16, %17) : (!torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>) -> !torch.vtensor<[?,?,1],f32> 
    %19 = torch.operator "onnx.Add"(%18, %1) : (!torch.vtensor<[?,?,1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[?,?,1],f32> 
    %20 = torch.operator "onnx.Sqrt"(%19) : (!torch.vtensor<[?,?,1],f32>) -> !torch.vtensor<[?,?,1],f32> 
    %21 = torch.operator "onnx.Sub"(%12, %14) : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,1],f32>) -> !torch.vtensor<[?,?,?],f32> 
    %22 = torch.operator "onnx.Div"(%21, %20) : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,1],f32>) -> !torch.vtensor<[?,?,?],f32> 
    %23 = torch.operator "onnx.Reshape"(%22, %3) : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[3,4,2,2],f32> 
    %24 = torch.operator "onnx.Reshape"(%23, %11) : (!torch.vtensor<[3,4,2,2],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,4,4],f32> 
    %25 = torch.operator "onnx.Cast"(%24) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[3,4,4],f32>) -> !torch.vtensor<[3,4,4],f32> 
    %26 = torch.operator "onnx.Constant"() {torch.onnx.value_ints = [1 : si64, -1 : si64, 1 : si64]} : () -> !torch.vtensor<[3],si64> 
    %27 = torch.operator "onnx.Cast"(%arg1) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> 
    %28 = torch.operator "onnx.Cast"(%arg2) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> 
    %29 = torch.operator "onnx.Reshape"(%27, %26) : (!torch.vtensor<[4],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,4,1],f32> 
    %30 = torch.operator "onnx.Reshape"(%28, %26) : (!torch.vtensor<[4],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,4,1],f32> 
    %31 = torch.operator "onnx.Mul"(%29, %25) : (!torch.vtensor<[1,4,1],f32>, !torch.vtensor<[3,4,4],f32>) -> !torch.vtensor<[3,4,4],f32> 
    %32 = torch.operator "onnx.Add"(%31, %30) : (!torch.vtensor<[3,4,4],f32>, !torch.vtensor<[1,4,1],f32>) -> !torch.vtensor<[3,4,4],f32> 
    %33 = torch.operator "onnx.Reshape"(%32, %3) : (!torch.vtensor<[3,4,4],f32>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[3,4,2,2],f32> 
    return %33 : !torch.vtensor<[3,4,2,2],f32>
  }
}

