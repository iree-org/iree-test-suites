#!/usr/bin/env python3
"""Regenerates the torchvision ViT-B/16 IREE quality-test artifacts.

Produces in --out-dir:
  model.mlir              exported torch-dialect MLIR
  real_weights.irpa       externalized parameters
  inference_input.0.bin   f32 pixel_values, shape [1, 3, 224, 224]
  inference_output.0.bin  f32 logits, shape [1, 1000]
"""

import argparse
import sys
from pathlib import Path

import torch
from torchvision.models import vit_b_16

import iree.turbine.aot as aot

SEED = 0
INPUT_SHAPE = (1, 3, 224, 224)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Creating torchvision vit_b_16 with random f32 weights (seed={SEED})",
        file=sys.stderr,
    )
    torch.manual_seed(SEED)
    model = vit_b_16(weights=None)
    model.eval()

    print(f"Creating random input tensor with shape {INPUT_SHAPE}", file=sys.stderr)
    pixel_values = torch.randn(*INPUT_SHAPE, dtype=torch.float32).contiguous()

    # Capture eager logits before externalizing params so numerics match the export.
    print("Running eager reference forward", file=sys.stderr)
    with torch.no_grad():
        ref_logits = model(pixel_values)
    ref_logits = ref_logits.to(torch.float32).contiguous()

    print("Writing reference input/output .bin files", file=sys.stderr)
    (args.out_dir / "inference_input.0.bin").write_bytes(
        pixel_values.numpy().astype("<f4").tobytes()
    )
    (args.out_dir / "inference_output.0.bin").write_bytes(
        ref_logits.numpy().astype("<f4").tobytes()
    )

    print("Exporting to MLIR", file=sys.stderr)
    aot.externalize_module_parameters(model)
    export_output = aot.export(model, args=(pixel_values,))
    export_output.save_mlir(str(args.out_dir / "model.mlir"))
    aot.save_module_parameters(str(args.out_dir / "real_weights.irpa"), model)

    print(f"Artifacts successfully written to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
