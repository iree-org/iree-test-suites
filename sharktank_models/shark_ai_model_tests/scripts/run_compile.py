'''
Compiles The Exported MLIR from Sharktank To vmfb

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
        subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Compile IR with IREE")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for dumping artifacts")
    parser.add_argument(
        "--extra-compile-flags-list",
        type=str,
        default="[]",
        help="Extra flags to pass as a Python-style list, e.g. '[\"--x\", \"--f\", \"--g\"]' or '[]'"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "../output_artifacts")
    os.makedirs(output_dir, exist_ok=True)

    print(" Compiling IR ....")
    start = time.time()

    compile_cmd = [
        "iree-compile",
        os.path.join(output_dir, "output.mlir"),
        "--iree-hip-target=gfx942",
        "-o", os.path.join(output_dir, "output.vmfb"),
        "--iree-opt-level=O3",
        "--iree-hal-indirect-command-buffers=true",
        "--iree-stream-resource-memory-model=discrete",
        "--iree-hip-enable-tensor-ukernels",
        "--iree-hal-memoization=true",
        "--iree-codegen-enable-default-tuning-specs=true",
        "--iree-stream-affinity-solver-max-iterations=1024",
        "--iree-hal-target-device=hip"
    ]

    try:
        extra_flags = ast.literal_eval(args.extra_compile_flags_list)
        if not isinstance(extra_flags, list):
            raise ValueError("Expected a list for --extra-compile-flags-list")
    except Exception as e:
        raise ValueError(f"Invalid value for --extra-compile-flags-list: {args.extra_compile_flags_list}") from e

    if len(extra_flags) == 0:
        print("No Extra Compile Flag is Passed")
    else:
        print("Appending Extra Compile Flags...")
        compile_cmd += extra_flags
        print("Command:", compile_cmd)

    run_command(compile_cmd)

    print(f"Time taken for compiling: {int(time.time() - start)} seconds")

if __name__ == "__main__":
    main()
