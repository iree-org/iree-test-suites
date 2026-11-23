# MobileNetV2 regression test

End-to-end regression for torchvision's
[MobileNetV2](https://docs.pytorch.org/vision/stable/models/mobilenetv2.html)
(`MobileNet_V2_Weights.IMAGENET1K_V2`) in two variants. IREE compiles the
committed MLIR, runs it on CPU, and compares the output logits against
PyTorch reference values fetched from the Hugging Face Hub.

| Variant | Description |
| --- | --- |
| `mobilenetv2-fp32` | The float32 model |
| `mobilenetv2-pt2e-xnnpack-static-w8_channel_sym-a8_tensor_asym` | PT2E post-training quantization with the ExecuTorch `XNNPACKQuantizer`: int8 weights, symmetric per output channel; int8 activations, asymmetric per tensor, calibrated on 64 random Imagenette training images |

## Files

| File | Description |
| --- | --- |
| `<variant>_quality_cpu.json` | Quality test definition |
| `modules/<variant>_cpu.json` | Module definition and compiler flags |
| `<variant>.mlir` | Exported torch-dialect program, parameters externalized |
| `generate.py` | Reproducer for the artifacts above and the parameters |
| `requirements.txt` | Pinned toolchain for `generate.py` |

Model parameters and reference inputs/outputs are hosted on the Hugging Face Hub
at [`roofline/iree-regression-models`](https://huggingface.co/roofline/iree-regression-models),
pinned by revision in the test definitions and fetched at test time:

| File | Description |
| --- | --- |
| `mobilenetv2/<variant>/real_weights.irpa` | Externalized model parameters |
| `mobilenetv2/<variant>/inference_input.0.bin` | Preprocessed `huggingface/cats-image` input sample (`1x3x224x224xf32`) |
| `mobilenetv2/<variant>/inference_output.0.bin` | Expected output logits (`1x1000xf32`) |

## Tolerances

The PT2E test allows 8 steps of the output's quantization scale
(`--expected_f32_threshold=0.489`), since tiny floating-point differences flip
the rounding of single values and the flips propagate. On the test image, IREE
is 5 steps off, and PyTorch differs from itself by 6 with another memory layout.

## Reproducing artifacts

```bash
pip install -r mobilenetv2/requirements.txt
python3 mobilenetv2/generate.py
```
