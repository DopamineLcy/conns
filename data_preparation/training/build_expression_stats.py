import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description="Build positive/negative expression statistics from extracted MIMIC JSON.")
    parser.add_argument("--reports-dir", default="data/conns_training/reports_extract_concepts")
    parser.add_argument("--metadata-csv", default="data/conns_training/mimic_conns_training.csv")
    parser.add_argument("--concepts-path", default="data/conns_training/concepts.json")
    parser.add_argument("--output-root", default="data/conns_training")
    parser.add_argument(
        "--segment-field",
        default="evidential_segment",
        help="JSON field used for expression text. The release prompt writes evidential_segment.",
    )
    return parser.parse_args()


def study_id_from_json(path):
    name = path.name
    if name.startswith("s") and name.endswith(".json"):
        return name[1:-5]
    return None


def collect_files(reports_dir, train_study_ids):
    files = []
    for root, _, names in os.walk(reports_dir):
        for name in names:
            if not name.endswith(".json"):
                continue
            path = Path(root) / name
            if study_id_from_json(path) in train_study_ids:
                files.append(path)
    return files


def save_counter(counter, label, presence, output_dir, total_files):
    if not counter:
        return None
    segments = counter.most_common()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{label.replace(' ', '_')}_segments_train.json"
    total_segments = sum(counter.values())
    payload = {
        "key": label,
        "presence": presence,
        "data_split": "train",
        "total_files_scanned": total_files,
        "total_segments": total_segments,
        "unique_segments": len(counter),
        "segments_by_frequency": [
            {"segment": segment, "count": count, "percentage": f"{count / total_segments * 100:.2f}%"}
            for segment, count in segments
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def as_segment(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return ""


def display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main():
    args = parse_args()
    reports_dir = repo_path(args.reports_dir)
    output_root = repo_path(args.output_root)
    yes_dir = output_root / "yes_expressions"
    no_dir = output_root / "no_expressions"
    yes_dir.mkdir(parents=True, exist_ok=True)
    no_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(repo_path(args.metadata_csv)).astype({"study_id": str, "split": str})
    train_study_ids = set(metadata[metadata["split"] == "train"]["study_id"])
    concepts = json.loads(repo_path(args.concepts_path).read_text(encoding="utf-8"))
    files = collect_files(reports_dir, train_study_ids)
    print(f"Using {len(files)} extracted train reports")
    if not files:
        raise RuntimeError(f"No train report JSON files found under {display_path(reports_dir)}")

    yes = {concept: Counter() for concept in concepts}
    no = {concept: Counter() for concept in concepts}
    missing_segment_field = 0
    missing_examples = []
    missing_example_set = set()
    for path in tqdm(files, total=len(files)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for concept in concepts:
            item = data.get(concept, {})
            presence = item.get("presence")
            if presence not in {"Yes", "No"}:
                continue
            if args.segment_field not in item:
                missing_segment_field += 1
                if len(missing_examples) < 3 and path not in missing_example_set:
                    missing_examples.append(path)
                    missing_example_set.add(path)
                continue
            segment = as_segment(item.get(args.segment_field, ""))
            if not segment:
                continue
            if presence == "Yes":
                yes[concept][segment] += 1
            else:
                no[concept][segment] += 1

    total_yes = sum(sum(counter.values()) for counter in yes.values())
    total_no = sum(sum(counter.values()) for counter in no.values())
    if total_yes + total_no == 0:
        if missing_segment_field:
            examples = ", ".join(display_path(path) for path in missing_examples)
            raise RuntimeError(
                f"No expressions were collected because {missing_segment_field} Yes/No entity entries "
                f"do not contain '{args.segment_field}'. Example files: {examples}. "
                "Regenerate extracted JSON with data_preparation/training/extract_entities.py "
                "using the current prompt, then run verify_extracted_json.py before rebuilding stats."
            )
        raise RuntimeError(
            f"No expressions were collected from {len(files)} train reports. "
            f"Check that '{args.segment_field}' is a non-empty string for Yes/No entities."
        )

    saved_yes = 0
    saved_no = 0
    for concept in concepts:
        if save_counter(yes[concept], concept, "Yes", yes_dir, len(files)):
            saved_yes += 1
        if save_counter(no[concept], concept, "No", no_dir, len(files)):
            saved_no += 1

    print(f"Collected {total_yes} Yes segments and {total_no} No segments")
    print(f"Saved {saved_yes} files under {display_path(yes_dir)}")
    print(f"Saved {saved_no} files under {display_path(no_dir)}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")
