# Torchvision ViT-B/16 

## Files

| File | Description |
| --- | --- |
| `torchvision-vit-b-16_random_benchmark_cpu.json` | Random-weight benchmark test definition |
| `modules/torchvision-vit-b-16_cpu.json` | Module definition and compiler flags |
| `model.mlir` | Exported torch-dialect program, parameters externalized |
| `generate.py` | Reproducer for the MLIR and future quality-test artifacts |
| `requirements.txt` | Pinned toolchain for `generate.py` |

The benchmark does not require hosted parameters or reference artifacts.

## Reproducing artifacts

`generate.py` can regenerate `model.mlir` and the future Hugging Face-hosted
quality-test artifacts:

```bash
pip install -r torchvision-vit-b-16/requirements.txt
python3 torchvision-vit-b-16/generate.py
```
