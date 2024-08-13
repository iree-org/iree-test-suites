module {
  func.func @test_loop11(%arg0: !torch.vtensor<[],si64>, %arg1: !torch.vtensor<[],i1>, %arg2: !torch.vtensor<[1],f32>) -> (!torch.vtensor<[1],f32>, !torch.vtensor<[5,1],f32>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:2 = torch.operator "onnx.Loop"(%arg0, %arg1, %arg2) : (!torch.vtensor<[],si64>, !torch.vtensor<[],i1>, !torch.vtensor<[1],f32>) -> (!torch.vtensor<[1],f32>, !torch.vtensor<[5,1],f32>) {
    ^bb0(%arg3: !torch.vtensor<[],si64>, %arg4: !torch.vtensor<[],i1>, %arg5: !torch.vtensor<[1],f32>):
      %1 = torch.operator "onnx.Identity"(%arg4) : (!torch.vtensor<[],i1>) -> !torch.vtensor<[],i1> 
      %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf32>} : () -> !torch.vtensor<[5],f32> 
      %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<si64>} : () -> !torch.vtensor<[],si64> 
      %4 = torch.operator "onnx.Add"(%arg3, %3) : (!torch.vtensor<[],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],si64> 
      %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
      %6 = torch.operator "onnx.Unsqueeze"(%arg3, %5) : (!torch.vtensor<[],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> 
      %7 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
      %8 = torch.operator "onnx.Unsqueeze"(%4, %7) : (!torch.vtensor<[],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> 
      %9 = torch.operator "onnx.Slice"(%2, %6, %8) : (!torch.vtensor<[5],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[?],f32> 
      %10 = torch.operator "onnx.Add"(%arg5, %9) : (!torch.vtensor<[1],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[1],f32> 
      %11 = torch.operator "onnx.Identity"(%10) : (!torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> 
      torch.operator_terminator %1, %10, %11 : !torch.vtensor<[],i1>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>
    }
    return %0#0, %0#1 : !torch.vtensor<[1],f32>, !torch.vtensor<[5,1],f32>
  }
}

