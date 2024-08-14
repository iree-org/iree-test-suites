module {
  func.func @test_affine_grid_2d_align_corners_expanded(%arg0: !torch.vtensor<[2,2,3],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[2,5,6,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value_int = 1 : si64} : () -> !torch.vtensor<[],si64> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value_int = 2 : si64} : () -> !torch.vtensor<[],si64> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value_int = 0 : si64} : () -> !torch.vtensor<[],si64> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value_int = 4 : si64} : () -> !torch.vtensor<[],si64> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value_ints = [1 : si64]} : () -> !torch.vtensor<[1],si64> 
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value_ints = [0 : si64]} : () -> !torch.vtensor<[1],si64> 
    %6 = torch.operator "onnx.Constant"() {torch.onnx.value_int = -1 : si64} : () -> !torch.vtensor<[],si64> 
    %7 = torch.operator "onnx.CastLike"(%6, %arg0) : (!torch.vtensor<[],si64>, !torch.vtensor<[2,2,3],f32>) -> !torch.vtensor<[],f32> 
    %8 = torch.operator "onnx.CastLike"(%2, %arg0) : (!torch.vtensor<[],si64>, !torch.vtensor<[2,2,3],f32>) -> !torch.vtensor<[],f32> 
    %9 = torch.operator "onnx.CastLike"(%0, %arg0) : (!torch.vtensor<[],si64>, !torch.vtensor<[2,2,3],f32>) -> !torch.vtensor<[],f32> 
    %10 = torch.operator "onnx.CastLike"(%1, %arg0) : (!torch.vtensor<[],si64>, !torch.vtensor<[2,2,3],f32>) -> !torch.vtensor<[],f32> 
    %11 = torch.operator "onnx.Constant"() {torch.onnx.value_int = 1 : si64} : () -> !torch.vtensor<[],si64> 
    %12 = torch.operator "onnx.Equal"(%11, %2) : (!torch.vtensor<[],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],i1> 
    %13 = torch.operator "onnx.Size"(%arg1) : (!torch.vtensor<[4],si64>) -> !torch.vtensor<[],si64> 
    %14 = torch.operator "onnx.Equal"(%13, %3) : (!torch.vtensor<[],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],i1> 
    %15:5 = torch.operator "onnx.If"(%14) : (!torch.vtensor<[],i1>) -> (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[?],si64>) {
      %63:5 = torch.operator "onnx.Split"(%arg1) {torch.onnx.num_outputs = 5 : si64} : (!torch.vtensor<[4],si64>) -> (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[0],si64>) 
      torch.operator_terminator %63#0, %63#1, %63#2, %63#3, %63#4 : !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[0],si64>
    }, {
      %63:4 = torch.operator "onnx.Split"(%arg1) {torch.onnx.num_outputs = 4 : si64} : (!torch.vtensor<[4],si64>) -> (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) 
      %64 = torch.operator "onnx.Identity"(%4) : (!torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> 
      torch.operator_terminator %63#0, %63#1, %64, %63#2, %63#3 : !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>
    }
    %16 = torch.operator "onnx.Concat"(%15#0, %15#1, %15#2, %15#3, %15#4) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[?],si64>) -> !torch.vtensor<[?],si64> 
    %17 = torch.operator "onnx.If"(%14) : (!torch.vtensor<[],i1>) -> !torch.vtensor<[],f32> {
      %63 = torch.operator "onnx.Identity"(%arg0) : (!torch.vtensor<[2,2,3],f32>) -> !torch.vtensor<[2,2,3],f32> 
      torch.operator_terminator %63 : !torch.vtensor<[2,2,3],f32>
    }, {
      %63 = torch.operator "onnx.Constant"() {torch.onnx.value_ints = [0 : si64, 1 : si64, 2 : si64, 0 : si64, 1 : si64, 2 : si64]} : () -> !torch.vtensor<[6],si64> 
      %64 = torch.operator "onnx.Constant"() {torch.onnx.value_ints = [2 : si64, 3 : si64]} : () -> !torch.vtensor<[2],si64> 
      %65 = torch.operator "onnx.Reshape"(%63, %64) : (!torch.vtensor<[6],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[2,3],si64> 
      %66 = torch.operator "onnx.Concat"(%15#0, %64) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[1],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[3],si64> 
      %67 = torch.operator "onnx.Expand"(%65, %66) : (!torch.vtensor<[2,3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[?,2,3],si64> 
      %68 = torch.operator "onnx.GatherElements"(%arg0, %67) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[2,2,3],f32>, !torch.vtensor<[?,2,3],si64>) -> !torch.vtensor<[?,2,3],f32> 
      %69:2 = torch.operator "onnx.Split"(%68) {torch.onnx.axis = 1 : si64, torch.onnx.num_outputs = 2 : si64} : (!torch.vtensor<[?,2,3],f32>) -> (!torch.vtensor<[?,1,3],f32>, !torch.vtensor<[?,1,3],f32>) 
      %70 = torch.operator "onnx.Squeeze"(%69#0) : (!torch.vtensor<[?,1,3],f32>) -> !torch.vtensor<[],f32> 
      %71 = torch.operator "onnx.Squeeze"(%69#1) : (!torch.vtensor<[?,1,3],f32>) -> !torch.vtensor<[],f32> 
      %72:3 = torch.operator "onnx.Split"(%70) {torch.onnx.axis = 1 : si64, torch.onnx.num_outputs = 3 : si64} : (!torch.vtensor<[],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) 
      %73:3 = torch.operator "onnx.Split"(%71) {torch.onnx.axis = 1 : si64, torch.onnx.num_outputs = 3 : si64} : (!torch.vtensor<[],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) 
      %74 = torch.operator "onnx.Shape"(%73#0) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[?],si64> 
      %75 = torch.operator "onnx.ConstantOfShape"(%74) : (!torch.vtensor<[?],si64>) -> !torch.vtensor<[],f32> 
      %76 = torch.operator "onnx.CastLike"(%75, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[2,2,3],f32>) -> !torch.vtensor<[],f32> 
      %77 = torch.operator "onnx.Add"(%76, %9) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
      %78 = torch.operator "onnx.Concat"(%72#0, %72#1, %76, %72#2) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
      %79 = torch.operator "onnx.Concat"(%73#0, %73#1, %76, %73#2) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
      %80 = torch.operator "onnx.Concat"(%76, %76, %77, %76) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
      %81 = torch.operator "onnx.Unsqueeze"(%78, %4) : (!torch.vtensor<[],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[],f32> 
      %82 = torch.operator "onnx.Unsqueeze"(%79, %4) : (!torch.vtensor<[],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[],f32> 
      %83 = torch.operator "onnx.Unsqueeze"(%80, %4) : (!torch.vtensor<[],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[],f32> 
      %84 = torch.operator "onnx.Concat"(%81, %82, %83) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
      torch.operator_terminator %84 : !torch.vtensor<[],f32>
    }
    %18 = torch.operator "onnx.Constant"() {torch.onnx.value_ints = [2 : si64]} : () -> !torch.vtensor<[1],si64> 
    %19 = torch.operator "onnx.Constant"() {torch.onnx.value_ints = [3 : si64]} : () -> !torch.vtensor<[1],si64> 
    %20 = torch.operator "onnx.Constant"() {torch.onnx.value_ints = [5 : si64]} : () -> !torch.vtensor<[1],si64> 
    %21 = torch.operator "onnx.Slice"(%16, %18, %20) : (!torch.vtensor<[?],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[?],si64> 
    %22 = torch.operator "onnx.ConstantOfShape"(%21) : (!torch.vtensor<[?],si64>) -> !torch.vtensor<[],f32> 
    %23 = torch.operator "onnx.CastLike"(%22, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[2,2,3],f32>) -> !torch.vtensor<[],f32> 
    %24 = torch.operator "onnx.Add"(%23, %9) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %25 = torch.operator "onnx.CastLike"(%15#2, %8) : (!torch.vtensor<[1],si64>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> 
    %26 = torch.operator "onnx.CastLike"(%15#3, %8) : (!torch.vtensor<[1],si64>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> 
    %27 = torch.operator "onnx.CastLike"(%15#4, %8) : (!torch.vtensor<[?],si64>, !torch.vtensor<[],f32>) -> !torch.vtensor<[?],f32> 
    %28:6 = torch.operator "onnx.If"(%12) : (!torch.vtensor<[],i1>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[?],f32>) {
      %63 = torch.operator "onnx.Sub"(%25, %9) : (!torch.vtensor<[1],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> 
      %64 = torch.operator "onnx.Sub"(%26, %9) : (!torch.vtensor<[1],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> 
      %65 = torch.operator "onnx.Sub"(%27, %9) : (!torch.vtensor<[?],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[?],f32> 
      %66 = torch.operator "onnx.Equal"(%15#2, %0) : (!torch.vtensor<[1],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[1],i1> 
      %67 = torch.operator "onnx.If"(%66) : (!torch.vtensor<[1],i1>) -> !torch.vtensor<[],f32> {
        %73 = torch.operator "onnx.Div"(%10, %63) : (!torch.vtensor<[],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> 
        torch.operator_terminator %73 : !torch.vtensor<[1],f32>
      }, {
        %73 = torch.operator "onnx.Identity"(%8) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
        torch.operator_terminator %73 : !torch.vtensor<[],f32>
      }
      %68 = torch.operator "onnx.Div"(%10, %64) : (!torch.vtensor<[],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> 
      %69 = torch.operator "onnx.Div"(%10, %65) : (!torch.vtensor<[],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> 
      %70 = torch.operator "onnx.Identity"(%7) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
      %71 = torch.operator "onnx.Identity"(%7) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
      %72 = torch.operator "onnx.Identity"(%7) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
      torch.operator_terminator %70, %67, %71, %68, %72, %69 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[?],f32>
    }, {
      %63 = torch.operator "onnx.Div"(%10, %25) : (!torch.vtensor<[],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> 
      %64 = torch.operator "onnx.Div"(%10, %26) : (!torch.vtensor<[],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> 
      %65 = torch.operator "onnx.Div"(%10, %27) : (!torch.vtensor<[],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> 
      %66 = torch.operator "onnx.Div"(%63, %10) : (!torch.vtensor<[1],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> 
      %67 = torch.operator "onnx.Add"(%7, %66) : (!torch.vtensor<[],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> 
      %68 = torch.operator "onnx.Div"(%64, %10) : (!torch.vtensor<[1],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> 
      %69 = torch.operator "onnx.Add"(%7, %68) : (!torch.vtensor<[],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> 
      %70 = torch.operator "onnx.Div"(%65, %10) : (!torch.vtensor<[?],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[?],f32> 
      %71 = torch.operator "onnx.Add"(%7, %70) : (!torch.vtensor<[],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> 
      torch.operator_terminator %67, %63, %69, %64, %71, %65 : !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>
    }
    %29 = torch.operator "onnx.Range"(%2, %15#4, %0) : (!torch.vtensor<[],si64>, !torch.vtensor<[?],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[?],si64> 
    %30 = torch.operator "onnx.CastLike"(%29, %28#5) : (!torch.vtensor<[?],si64>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> 
    %31 = torch.operator "onnx.Mul"(%30, %28#5) : (!torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> 
    %32 = torch.operator "onnx.Add"(%28#4, %31) : (!torch.vtensor<[],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[],f32> 
    %33 = torch.operator "onnx.Range"(%2, %15#3, %0) : (!torch.vtensor<[],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[?],si64> 
    %34 = torch.operator "onnx.CastLike"(%33, %28#3) : (!torch.vtensor<[?],si64>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[?],f32> 
    %35 = torch.operator "onnx.Mul"(%34, %28#3) : (!torch.vtensor<[?],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[?],f32> 
    %36 = torch.operator "onnx.Add"(%28#2, %35) : (!torch.vtensor<[],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[],f32> 
    %37 = torch.operator "onnx.Range"(%2, %15#2, %0) : (!torch.vtensor<[],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[?],si64> 
    %38 = torch.operator "onnx.CastLike"(%37, %28#1) : (!torch.vtensor<[?],si64>, !torch.vtensor<[],f32>) -> !torch.vtensor<[?],f32> 
    %39 = torch.operator "onnx.Mul"(%38, %28#1) : (!torch.vtensor<[?],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %40 = torch.operator "onnx.Add"(%28#0, %39) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %41 = torch.operator "onnx.Transpose"(%23) {torch.onnx.perm = [1 : si64, 2 : si64, 0 : si64]} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %42 = torch.operator "onnx.Add"(%41, %40) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %43 = torch.operator "onnx.Transpose"(%42) {torch.onnx.perm = [2 : si64, 0 : si64, 1 : si64]} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %44 = torch.operator "onnx.Transpose"(%23) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64]} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %45 = torch.operator "onnx.Add"(%44, %36) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %46 = torch.operator "onnx.Transpose"(%45) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64]} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %47 = torch.operator "onnx.Add"(%32, %23) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %48 = torch.operator "onnx.Unsqueeze"(%47, %6) : (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> 
    %49 = torch.operator "onnx.Unsqueeze"(%46, %6) : (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> 
    %50 = torch.operator "onnx.Unsqueeze"(%43, %6) : (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> 
    %51 = torch.operator "onnx.Unsqueeze"(%24, %6) : (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> 
    %52 = torch.operator "onnx.Concat"(%48, %49, %50, %51) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %53 = torch.operator "onnx.Constant"() {torch.onnx.value_ints = [-1 : si64, 4 : si64]} : () -> !torch.vtensor<[2],si64> 
    %54 = torch.operator "onnx.Reshape"(%52, %53) : (!torch.vtensor<[],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[?,4],f32> 
    %55 = torch.operator "onnx.Transpose"(%54) : (!torch.vtensor<[?,4],f32>) -> !torch.vtensor<[4,?],f32> 
    %56 = torch.operator "onnx.CastLike"(%55, %17) : (!torch.vtensor<[4,?],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,?],f32> 
    %57 = torch.operator "onnx.MatMul"(%17, %56) : (!torch.vtensor<[],f32>, !torch.vtensor<[4,?],f32>) -> !torch.vtensor<[],f32> 
    %58 = torch.operator "onnx.Transpose"(%57) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64]} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %59 = torch.operator "onnx.Concat"(%15#0, %15#2, %15#3, %15#4, %19) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[?],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[?],si64> 
    %60 = torch.operator "onnx.Reshape"(%58, %59) : (!torch.vtensor<[],f32>, !torch.vtensor<[?],si64>) -> !torch.vtensor<[],f32> 
    %61 = torch.operator "onnx.CastLike"(%60, %17) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %62 = torch.operator "onnx.If"(%14) : (!torch.vtensor<[],i1>) -> !torch.vtensor<[2,5,6,2],f32> {
      %63 = torch.operator "onnx.Identity"(%61) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
      torch.operator_terminator %63 : !torch.vtensor<[],f32>
    }, {
      %63 = torch.operator "onnx.Squeeze"(%61, %4) : (!torch.vtensor<[],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[],f32> 
      %64 = torch.operator "onnx.Slice"(%63, %5, %18, %19) : (!torch.vtensor<[],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[],f32> 
      torch.operator_terminator %64 : !torch.vtensor<[],f32>
    }
    return %62 : !torch.vtensor<[2,5,6,2],f32>
  }
}

