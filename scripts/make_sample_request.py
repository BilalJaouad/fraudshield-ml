"""Create an API request from an original Base.csv observation (never from preprocessed data)."""
import argparse
import json
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=Path, required=True, help="Path to raw Base.csv")
parser.add_argument("--row", type=int, default=0)
args = parser.parse_args()
with Path("artifacts/feature_schema.json").open(encoding="utf-8") as f:
    schema = json.load(f)
raw = pd.read_csv(args.csv, usecols=schema["raw_features"])
record = raw.iloc[args.row].to_dict()
Path("sample_request.json").write_text(json.dumps({"features": record}, indent=2), encoding="utf-8")
print("Generated sample_request.json — paste its contents into POST /predict in /docs")
