module {
  func.func @test_ai_onnx_ml_tree_ensemble_single_tree(%arg0: !torch.vtensor<[3,2],f64>) -> !torch.vtensor<[3,2],f64> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 5 : si64}, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.TreeEnsemble"(%arg0) {torch.onnx.aggregate_function = 1 : si64, torch.onnx.leaf_targetids = [0 : si64, 1 : si64, 0 : si64, 1 : si64], torch.onnx.leaf_weights = dense<[5.230000e+00, 1.212000e+01, -1.223000e+01, 7.210000e+00]> : tensor<4xf64>, torch.onnx.n_targets = 2 : si64, torch.onnx.nodes_falseleafs = [0 : si64, 1 : si64, 1 : si64], torch.onnx.nodes_falsenodeids = [2 : si64, 2 : si64, 3 : si64], torch.onnx.nodes_featureids = [0 : si64, 0 : si64, 0 : si64], torch.onnx.nodes_modes = dense<0> : tensor<3xui8>, torch.onnx.nodes_splits = dense<[3.140000e+00, 1.200000e+00, 4.200000e+00]> : tensor<3xf64>, torch.onnx.nodes_trueleafs = [0 : si64, 1 : si64, 1 : si64], torch.onnx.nodes_truenodeids = [1 : si64, 0 : si64, 1 : si64], torch.onnx.post_transform = 0 : si64, torch.onnx.tree_roots = [0 : si64]} : (!torch.vtensor<[3,2],f64>) -> !torch.vtensor<[3,2],f64> 
    return %0 : !torch.vtensor<[3,2],f64>
  }
}

