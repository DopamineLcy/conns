import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description="Create PadChest-GR evaluation CSV/JSON files.")
    parser.add_argument("--raw-root", default="data/PadChest-GR")
    parser.add_argument("--output-root", default="data/conns_evaluation/PadChest-GR")
    return parser.parse_args()


def create_classification(master_table, output_path):
    df = pd.read_csv(master_table)
    test_df = df[df["split"] == "test"].copy()
    binary = pd.crosstab(test_df["ImageID"], test_df["label"])
    binary = (binary > 0).astype(int).reset_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary.to_csv(output_path, index=False)
    print(f"Saved classification CSV: {output_path}")


def create_grounding(master_table, reports_json, output_path):
    df = pd.read_csv(master_table)
    test_ids = set(df[df["split"] == "test"]["ImageID"])
    valid_labels = {
        "NSG tube", "alveolar pattern", "aortic atheromatosis", "aortic elongation",
        "atelectasis", "bronchiectasis", "cardiomegaly", "central venous catheter",
        "electrical device", "endotracheal tube", "fracture", "goiter",
        "hemidiaphragm elevation", "hiatal hernia", "hyperinflated lung", "hypoexpansion",
        "interstitial pattern", "nodule", "osteopenia", "pleural effusion",
        "pleural thickening", "scoliosis", "vascular hilar enlargement",
        "vertebral degenerative changes",
    }
    reports = json.loads(reports_json.read_text(encoding="utf-8"))
    rows = []
    for report in reports:
        image_id = report.get("ImageID")
        if image_id not in test_ids:
            continue
        for finding in report.get("findings", []):
            if finding.get("abnormal") is not True:
                continue
            labels = finding.get("labels") or []
            if not labels or labels[0] not in valid_labels:
                continue
            rows.append({"image": image_id, "text": finding.get("sentence_en"), "bbox": finding.get("boxes")})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved grounding JSON: {output_path}")


def main():
    args = parse_args()
    raw_root = repo_path(args.raw_root)
    output_root = repo_path(args.output_root)
    master_table = raw_root / "master_table.csv"
    reports_json = raw_root / "grounded_reports_20240819.json"
    if not master_table.exists():
        raise FileNotFoundError(f"missing {master_table}")
    if not reports_json.exists():
        raise FileNotFoundError(f"missing {reports_json}")
    create_classification(master_table, output_root / "test_binary_classification.csv")
    create_grounding(master_table, reports_json, output_root / "test_grounding.json")


if __name__ == "__main__":
    main()
