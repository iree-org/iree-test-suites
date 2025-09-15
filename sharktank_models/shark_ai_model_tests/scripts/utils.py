'''
Combines the IREE Benchmark Reports for Prefill and Decode into a Single
File -> consolidated_benchmark.json

Then Tests the Results Through Prefill and Decode Time 
By Comparing with a Gold Number
Tolerance 3% for Prefill and 6% for Decode
'''
import json
import argparse
from pathlib import Path
import numpy as np
import sys
import glob


def combine_json(dir, outfile):
    files = glob.glob(str(dir.absolute()) + "/*.json")
    merged_data = [json.load(open(path, "r")) for path in files]
    with open(outfile, "w") as outs:
        json.dump(merged_data, outs, indent=2)

def append_isl_to_json(dir, isl=None):
    files = glob.glob(str(dir.absolute()) + "/*.json")
    for f in files:
        length = isl
        if not length:
            length = Path(f).stem.rsplit("isl_")[-1]
        try:
            length = int(length)
        except Exception as e:
            print(f"Invalid ITL encountered, Exception {e}")

        with open(f, "r") as src:
            data = json.load(src)
            if "context" in data:
                context = data["context"]
                context["ISL"] = length

                with open(f, "w") as src:
                    json.dump(data, src, indent=2)


# ############ Test Above Results Through Prefill-Decode Time ################
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combine-json",
        type=Path,
        help="Combine all json files into single file",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output json file name",
    )
    parser.add_argument(
        "--append-isl",
        action="store_true",
        help="Append isl to the json",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=None,
        help="Input sequence length to append to the json",
    )
    args = parser.parse_args()

    if args.append_isl:
        append_isl_to_json(args.combine_json, args.isl)
    combine_json(args.combine_json, args.output_json)


    ##### Test for Prefill and Decode Time #####
    VERY_LARGE = 1e9
    ISL = [2048]
    metrics = extract_prefill_decode_pairs_for_isl(, ISL, model)
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
