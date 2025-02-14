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

for filename in os.listdir(f"./sharktank_models/test_suite/regression_tests/{model}"):
    if ".json" in filename:
        submodel_name = filename.split(".")[0]
        command = [
            "pytest",
            "./sharktank_models/test_suite/regression_tests/test_model_threshold.py",
            "-rpFe",
            "--log-cli-level=info",
            "--capture=no",
            "--timeout=600",
            "--durations=0",
            f"--model-name={model}",
            f"--submodel-name={submodel_name}"
        ]
        subprocess.run(command)