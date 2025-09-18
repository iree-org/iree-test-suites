'''
Exports The MLIR from Sharktank

'''

import argparse
import os
import subprocess
import sys
import time
import ast

def run_command(cmd, **kwargs):
    print(f"Running: {' '.join(cmd)}")
    try:
        start = time.time()
        subprocess.run(cmd, check=True, **kwargs)
        print(f"Time taken for exporting: {int(time.time() - start)} seconds")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Export IR with IREE")
    parser.add_argument("--irpa", default="/shark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa",
                        help="Path to IRPA file")
    parser.add_argument("--bs-prefill", default="1,2,4,8", help="Prefill batch sizes")
    parser.add_argument("--bs-decode", default="4,8,16,32,64", help="Decode batch sizes")
    parser.add_argument("--dtype", default="fp16", help="Data type (fp16/fp8/mistral_fp8)")
    parser.add_argument("--attention-kernel", default="sharktank", help="Which Attention Kernel To Use")
    parser.add_argument("--device-block-count", default="4096", help="What Device Block Count To Be Used")
    parser.add_argument(
    "--extra-export-flags-list",
    type=str,
    default="[]",
    help="Extra flags to pass as a Python-style list, e.g. '[\"--x\", \"--f\", \"--g\"]' or '[]'"
)
    parser.add_argument("--output-dir", default="output_artifacts/",
                        help="Output directory for dumping artifacts")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "../output_artifacts")
    os.makedirs(output_dir, exist_ok=True)


    os.environ["OUTPUT_DIR"] = args.output_dir
    os.environ["IRPA_PATH"] = args.irpa
    if args.dtype == "fp8":
        os.environ["ATTENTION_DTYPE"] = "float16"
        os.environ["ACTIVATION_DTYPE"] = "float16"
        os.environ["KV_CACHE_DTYPE"] = "float8_e4m3fnuz"

    ### Starting Export ###
    print("Exporting IR ....")


    export_cmd = [
        sys.executable, "-m", "sharktank.examples.export_paged_llm_v1",
        f"--irpa-file={args.irpa}",
        f"--output-mlir={os.path.join(output_dir, 'output.mlir')}",
        f"--output-config={os.path.join(output_dir, 'config_attn.json')}",
        f"--bs-prefill={args.bs_prefill}",
        f"--bs-decode={args.bs_decode}",
        f"--attention-kernel={args.attention_kernel}",
        "--use-hf",
        f"--device-block-count", f"{args.device_block_count} "
        # "--attention-dtype", "float16",
        # "--activation-dtype", "float16",
        # "--kv-cache-dtype", "float8_e4m3fnuz"
    ]


    try:
        extra_flags = ast.literal_eval(args.extra_export_flags_list)
        if not isinstance(extra_flags, list):
            raise ValueError("Expected a list for --extra-export-flags-list")
    except Exception as e:
        raise ValueError(f"Invalid value for --extra-export-flags-list: {args.extra_export_flags_list}") from e


    if len(extra_flags) == 0:
        print("No Extra Export Flag is Passed")
    else:
        print("Appending Extra Export Flags...")
        export_cmd += extra_flags
        print("Command:", export_cmd)


    run_command(export_cmd)

if __name__ == "__main__":
    main()
