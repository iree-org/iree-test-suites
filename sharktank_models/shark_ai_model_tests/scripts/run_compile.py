'''
Compiles The Exported MLIR from Sharktank To vmfb file

'''
import argparse
import os
import subprocess
import sys
import time

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
    parser.add_argument("--prefill-gold", required=True, default=None,
                        help="Gold Number for Prefill: Tolerance -> 3%")
    parser.add_argument("--decode-gold", required=True, default=None,
                        help="Gold Number for Decode: Tolerance -> 6%)")

    args = parser.parse_args()
    
    
    gold_number=args.gold_number
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
        "--iree-codegen-enable-default-tuning-specs=true"
    ]

    compile_cmd += ["--iree-hal-target-device=hip"]
    run_command(compile_cmd)

    print(f"Time taken for compiling: {int(time.time() - start)} seconds")

if __name__ == "__main__":
    main()




# ############ Test Results Through Prefill-Decode Time ################

ISL = [2048]

def extract_prefill_decode_pairs_for_isl(json_path, target_isl, model):
    with open(json_path, "r") as f:
        data = json.load(f)

    results = []
    prefill_map = {}
    decode_map = {}
    for entry in data:
        context = entry.get("context", {})
        isl = context.get("ISL")
        if isl != target_isl:
            continue

        for bench in entry.get("benchmarks", []):
            name = bench.get("name", "")
            run_type = bench.get("run_type", "")
            if run_type != "aggregate" or "mean" not in name:
                continue

            bs_match = re.search(r'bs(\d+)', name)
            if not bs_match:
                continue
            bs = int(bs_match.group(1))

            if "prefill" in name:
                prefill_map[bs] = round(bench.get("real_time", VERY_LARGE), 3)
            elif "decode" in name:
                decode_map[bs] = round(bench.get("real_time", VERY_LARGE), 3)

    for prefill_bs, prefill_time in sorted(prefill_map.items()):
        if prefill_bs != 4:
            continue
        decode_bs = prefill_bs * 8 if model == "Mistral-Nemo-Instruct-2407-FP8" else prefill_bs
        decode_time = decode_map.get(decode_bs, VERY_LARGE)

    results.append({
         "prefill_batch_size": prefill_bs,
         "Today's Prefill Time(ms)": prefill_time,
         "decode_batch_size": decode_bs,
         "Today's Decode Time(ms)": decode_time,
         "ISL": isl
        })

    return results


def prefill_status(current, historical): # 3% tolerance
    if current == "-":
        return "PASS"
    if pd.isna(historical) or historical == "-":
        return "PASS"
    return "PASS" if current <= 1.03 * float(historical) else "FAIL"  # 6% tolerance

def decode_status(current, historical):
    if current == "-"
    	return "PASS"
    if pd.isna(historical) or historical == "-":
  	return "PASS"
    return "PASS" if current <= 1.06 * float(historical) else "FAIL"


metrics = extract_prefill_decode_pairs_for_isl(json_path, ISL, model)
metrics.sort(key=lambda x: x['prefill_batch_size'])

prefill_status = "FAILED"
decode_status  = "FAILED"
for data in metrics:
 	prefill_status = "-" if metrics[0] == VERY_LARGE else prefill_status(data["Today's Prefill Time(ms)"], args.prefill_gold )
    decode_status= "-" if metrics[0] == VERY_LARGE else decode_status(data["Today's Decode Time(ms)"], args.decode_gold)


if prefill_status == "PASS" and decode_status == "PASS":
        print("[SUCCESS] Both prefill and decode status are within 3% and 6% of tolerance w.r.t the Gold Number")
elif prefill_status == "FAIL" and decode_status == "PASS":
 	print("[FAIL] Prefill Number Not within 3% tolerance of Gold number")
 	exit 1
elif prefill_status == "PASS" and decode_status == "FAIL":
 	print("[FAIL] Decode Number Not within 6% tolerance of Gold Number")
        exit 1
else:
 	print("[FAIL] Both docode and prefill not within range of their respective 3% and 6% tolerance.")
 	exit 1
