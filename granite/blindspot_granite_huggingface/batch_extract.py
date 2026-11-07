"""Process a small CSV sample with the hosted Granite model.

Usage: python batch_extract.py input.csv --limit 10
Requires cve_id and description columns. Outputs are written to outputs/.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from granite_api import extract_vulnerability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path).fillna("")
    if not {"cve_id", "description"}.issubset(df.columns):
        raise ValueError("CSV must have cve_id and description columns")
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    rows = []
    for _, record in df.head(args.limit).iterrows():
        cve_id = str(record["cve_id"])
        try:
            extracted = extract_vulnerability(str(record["description"]))
            rows.append({"cve_id": cve_id, "error": "", **extracted})
            print("OK", cve_id)
        except Exception as exc:
            rows.append({"cve_id": cve_id, "error": str(exc)})
            print("ERROR", cve_id, str(exc))
    pd.DataFrame(rows).to_csv(out / "granite_features.csv", index=False)
    print("Saved outputs/granite_features.csv")


if __name__ == "__main__":
    main()
