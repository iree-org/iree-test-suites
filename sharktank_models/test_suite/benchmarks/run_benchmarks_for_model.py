import subprocess
import  os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str)
args = parser.parse_args()
model = args.model

"""
This is temporary, you could do this in the GH action in order to display each actual test: https://github.com/orgs/community/discussions/58007

Will evaluate during GH action implementation
"""

for filename in os.listdir(f"./sharktank_models/test_suite/benchmarks/{model}"):
    if ".json" in filename:
        submodel_name = filename.split(".")[0]
        command = [
            "pytest",
            "./sharktank_models/test_suite/benchmarks/test_model_benchmark.py",
            "--log-cli-level=info",
            "--timeout=600",
            "--retries=7",
            f"--model-name={model}",
            f"--submodel-name={submodel_name}"
        ]
        subprocess.run(command)