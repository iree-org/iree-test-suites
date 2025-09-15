import argparse
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd: list[str]):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run IREE Benchmark.")
    parser.add_argument("--parameters", required=True, help="Path to IRPA file")
    parser.add_argument("--vmfb", default=None, help="Path to VMFB file")
    parser.add_argument("--model", required=True, help="Model name")
    # parser.add_argument("--bs-prefill", default="1,2,4,8", help="Prefill batch sizes (default: 1,2,4,8)")
    # parser.add_argument("--bs-decode", default="4,8,16,32,64", help="Decode batch sizes (default: 4,8,16,32,64)")
    parser.add_argument("--extra-compile-flags", default=[], help="Add Extra Flags That Have To Be Passed For a Specific Model in test/configs")


    parser.add_argument("--benchmarks", required=True, help="(see format in ../tests/configs.py file):<benchmark_name>, [<comma seperated input values>], <ISL>")
    parser.add_argument("--benchmark-repetation", required=True, help="eg: 3 (see format in ../tests/configs.py file): ")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parent / "output_artifacts"
    vmfb = args.vmfb or str(output_dir / "output.vmfb")
    benchmark_dir = output_dir / "benchmark_module"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    irpa_path = args.parameters
    model = args.model
    print(f"Model: {model}")

    benchmarks = args.benchmarks
    for func, inputs, isl in benchmarks:
        out_file = benchmark_dir / f"{model}_{func}_isl_{isl}.json"
        print(f"Running {model} {func} ISL: {isl}")
        compile_command = [
            "iree-benchmark-module",
            "--device_allocator=caching",
            "--hip_use_streams=true",
            f"--module={vmfb}",
            f"--parameters=model={irpa_path}",
            "--device=hip",
            f"--function={func}",
            # *[f"--input={i}" for i in inputs.split()],
            *[f"--input={i}" for i in inputs],
            f"--benchmark_repetitions={args.benchmark_repetition}",
            "--benchmark_out_format=json",
            f"--benchmark_out={out_file}",
        ]

        compile_command += extra_compile_flags
        run_cmd(compile_command)
      
      
if __name__ == "__main__":
    main()
