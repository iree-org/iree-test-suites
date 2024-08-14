module {
  func.func @test_ai_onnx_ml_tree_ensemble_set_membership(%arg0: !torch.vtensor<[6,1],f32>) -> !torch.vtensor<[6,4],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 5 : si64}, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.TreeEnsemble"(%arg0) {torch.onnx.aggregate_function = 1 : si64, torch.onnx.leaf_targetids = [0 : si64, 1 : si64, 2 : si64, 3 : si64], torch.onnx.leaf_weights = dense<[1.000000e+00, 1.000000e+01, 1.000000e+03, 1.000000e+02]> : tensor<4xf32>, torch.onnx.membership_values = dense<[1.200000e+00, 3.700000e+00, 8.000000e+00, 9.000000e+00, 0x7FC00000, 1.200000e+01, 7.000000e+00, 0x7FC00000]> : tensor<8xf32>, torch.onnx.n_targets = 4 : si64, torch.onnx.nodes_falseleafs = [1 : si64, 0 : si64, 1 : si64], torch.onnx.nodes_falsenodeids = [2 : si64, 2 : si64, 3 : si64], torch.onnx.nodes_featureids = [0 : si64, 0 : si64, 0 : si64], torch.onnx.nodes_modes = dense<[0, 6, 6]> : tensor<3xui8>, torch.onnx.nodes_splits = dense<[1.100000e+01, 2.323440e+05, 0x7FC00000]> : tensor<3xf32>, torch.onnx.nodes_trueleafs = [0 : si64, 1 : si64, 1 : si64], torch.onnx.nodes_truenodeids = [1 : si64, 0 : si64, 1 : si64], torch.onnx.post_transform = 0 : si64, torch.onnx.tree_roots = [0 : si64]} : (!torch.vtensor<[6,1],f32>) -> !torch.vtensor<[6,4],f32> 
    return %0 : !torch.vtensor<[6,4],f32>
  }
}

