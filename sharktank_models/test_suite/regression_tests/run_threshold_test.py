import subprocess
import  os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="sdxl")
parser.add_argument("-s", "--submodel", type=str, default="*")
args = parser.parse_args()
model = args.model
submodel = args.submodel

os.environ['THRESHOLD_MODEL'] = model
os.environ['THRESHOLD_SUBMODEL'] = submodel

command = [
    "pytest",
    f"{Path.cwd()}/sharktank_models/test_suite/regression_tests/test_model_threshold.py",
    "-rpFe",
    "--log-cli-level=info",
    "--capture=no",
    "--timeout=600",
    "--durations=0"
]
subprocess.run(command)