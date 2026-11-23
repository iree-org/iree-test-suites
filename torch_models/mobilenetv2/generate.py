#!/usr/bin/env python3
"""Regenerates the MobileNetV2 IREE quality-test artifacts.

Produces in --out-dir for each variant:
  <variant>.mlir                    exported torch-dialect MLIR
  <variant>/real_weights.irpa       externalized parameters
  <variant>/inference_input.0.bin   f32 pixel_values, shape [1, 3, 224, 224]
  <variant>/inference_output.0.bin  f32 logits, shape [1, 1000]
"""

import argparse
import sys
from pathlib import Path

import iree.turbine.aot as aot
import torch
import torchvision
from datasets import load_dataset
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

MODEL_WEIGHTS = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
SAMPLE_DATASET_ID = "huggingface/cats-image"
SAMPLE_DATASET_REVISION = "4613f5f1d3642cc2d56ffdf1b58c9d0f912cdc1f"
CALIBRATION_SAMPLES = 64
CALIBRATION_SEED = 0

XNNPACK_QUANTIZER = XNNPACKQuantizer().set_global(
    get_symmetric_quantization_config(is_per_channel=True)
)

# Maps each model variant to its PT2E quantizer (None for the float model).
# The variant name is also its output and Hugging Face folder name.
MODEL_VARIANTS = {
    "mobilenetv2-fp32": None,
    "mobilenetv2-pt2e-xnnpack-static-w8_channel_sym-a8_tensor_asym": XNNPACK_QUANTIZER,
}


def quantize_pt2e(
    model, quantizer, example_input, calibration_dir
) -> torch.fx.GraphModule:
    """PT2E post-training quantization, calibrated on Imagenette images."""
    print(
        f"Calibrating on {CALIBRATION_SAMPLES} Imagenette training images",
        file=sys.stderr,
    )
    dataset = torchvision.datasets.Imagenette(
        calibration_dir,
        split="train",
        size="320px",
        download=True,
        transform=MODEL_WEIGHTS.transforms(),
    )
    calibration_set, _ = torch.utils.data.random_split(
        dataset,
        [CALIBRATION_SAMPLES, len(dataset) - CALIBRATION_SAMPLES],
        generator=torch.Generator().manual_seed(CALIBRATION_SEED),
    )
    model = prepare_pt2e(
        torch.export.export(model, (example_input,)).module(), quantizer
    )
    with torch.no_grad():
        for image, _ in calibration_set:
            model(image.unsqueeze(0))
    model = convert_pt2e(model)
    # Keep only the attributes the graph uses. The converted module still holds
    # the float weights, which would otherwise end up in the .irpa.
    return torch.fx.GraphModule(model, model.graph)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts",
    )
    args = parser.parse_args()

    print(
        "Loading and preprocessing sample image from "
        f"{SAMPLE_DATASET_ID}@{SAMPLE_DATASET_REVISION}",
        file=sys.stderr,
    )
    image = load_dataset(SAMPLE_DATASET_ID, revision=SAMPLE_DATASET_REVISION)["test"][
        "image"
    ][0]
    # The transform returns a channels-last view. Make it contiguous so the eager
    # reference sees the same NCHW memory layout as the .bin input.
    pixel_values = (
        MODEL_WEIGHTS.transforms()(image.convert("RGB")).unsqueeze(0).contiguous()
    )

    for model_variant_name, quantizer in MODEL_VARIANTS.items():
        variant_dir = args.out_dir / model_variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Loading model torchvision {MODEL_WEIGHTS} for {model_variant_name}",
            file=sys.stderr,
        )
        model = torchvision.models.mobilenet_v2(weights=MODEL_WEIGHTS).eval()
        if quantizer is not None:
            model = quantize_pt2e(model, quantizer, pixel_values, args.out_dir)

        # Capture eager logits before externalizing params so numerics match the export.
        print("Running eager reference forward", file=sys.stderr)
        with torch.no_grad():
            ref_logits = model(pixel_values)

        # Dump raw reference input/output data.
        print("Writing reference input/output .bin files", file=sys.stderr)
        (variant_dir / "inference_input.0.bin").write_bytes(
            pixel_values.numpy().astype("<f4").tobytes()
        )
        (variant_dir / "inference_output.0.bin").write_bytes(
            ref_logits.numpy().astype("<f4").tobytes()
        )

        # Export to MLIR with weights externalized into a separate .irpa.
        print("Exporting to MLIR", file=sys.stderr)
        aot.externalize_module_parameters(model)
        export_output = aot.export(model, args=(pixel_values,))
        export_output.save_mlir(str(args.out_dir / f"{model_variant_name}.mlir"))
        aot.save_module_parameters(str(variant_dir / "real_weights.irpa"), model)

        print(f"Artifacts successfully written to {variant_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
