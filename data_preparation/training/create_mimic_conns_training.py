import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description="Create CoNNS MIMIC-CXR training CSV.")
    parser.add_argument("--mimic-root", default="data/raw_dataset/MIMIC-CXR-JPG")
    parser.add_argument("--ms-cxr-csv", default="data/MS-CXR/MS_CXR_Local_Alignment_v1.1.0.csv")
    parser.add_argument("--output", default="data/conns_training/mimic_conns_training.csv")
    parser.add_argument("--frontal-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    mimic_root = repo_path(args.mimic_root)
    metadata_csv = mimic_root / "mimic-cxr-2.0.0-metadata.csv.gz"
    split_csv = mimic_root / "mimic-cxr-2.0.0-split.csv.gz"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata CSV not found: {metadata_csv}")
    if not split_csv.exists():
        raise FileNotFoundError(f"split CSV not found: {split_csv}")

    df = pd.read_csv(metadata_csv).astype({"subject_id": "string", "study_id": "string", "dicom_id": "string"})
    split_df = pd.read_csv(split_csv).astype({"subject_id": "string", "study_id": "string", "dicom_id": "string"})

    if args.frontal_only:
        df = df[df["ViewPosition"].isin(["AP", "PA"])]
        df = df.drop_duplicates(subset=["study_id"], keep=False)

    split_unique = split_df.drop_duplicates(subset=["study_id"])
    df["split"] = df["study_id"].map(split_unique.set_index("study_id")["split"])
    df = df.dropna(subset=["split"])
    df = df[["dicom_id", "subject_id", "study_id", "split"]]

    ms_cxr_csv = repo_path(args.ms_cxr_csv)
    if not ms_cxr_csv.exists():
        raise FileNotFoundError(f"MS-CXR alignment CSV not found: {ms_cxr_csv}")
    ms_cxr_df = pd.read_csv(ms_cxr_csv)
    if "path" not in ms_cxr_df.columns:
        raise ValueError(f"MS-CXR alignment CSV must contain a path column: {ms_cxr_csv}")
    ms_cxr_subject_ids = ms_cxr_df["path"].astype(str).str.split("/").str[2].str.replace("p", "", regex=False)
    before = len(df)
    df = df[~df["subject_id"].isin(set(ms_cxr_subject_ids.dropna()))]
    removed = before - len(df)

    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Removed {removed} MIMIC rows overlapping MS-CXR subjects")
    print(f"Saved {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
