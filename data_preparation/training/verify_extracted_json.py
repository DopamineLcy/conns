import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FIELDS = {"presence", "location", "evidential_segment", "characteristics"}
ALLOWED_PRESENCE = {"Yes", "No", "Not Mentioned", "Uncertain"}


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description="Verify extracted MIMIC entity JSON files.")
    parser.add_argument("--input-dir", default="data/conns_training/reports_extract_concepts")
    parser.add_argument("--concepts-path", default="data/conns_training/concepts.json")
    parser.add_argument("--failed-list", default="data/conns_training/failed_extracted_json.txt")
    return parser.parse_args()


def validate(data, concepts):
    if not isinstance(data, dict):
        return False, "root is not an object"
    missing = set(concepts) - set(data)
    if missing:
        return False, f"missing concepts: {sorted(missing)[:5]}"
    for concept in concepts:
        item = data[concept]
        if not isinstance(item, dict):
            return False, f"{concept} is not an object"
        missing_fields = REQUIRED_FIELDS - set(item)
        if missing_fields:
            return False, f"{concept} missing fields: {sorted(missing_fields)}"
        if item["presence"] not in ALLOWED_PRESENCE:
            return False, f"{concept} invalid presence: {item['presence']}"
        if not isinstance(item["characteristics"], list):
            return False, f"{concept} characteristics is not a list"
    return True, "ok"


def main():
    args = parse_args()
    input_dir = repo_path(args.input_dir)
    concepts = json.loads(repo_path(args.concepts_path).read_text(encoding="utf-8"))
    files = []
    for root, _, names in os.walk(input_dir):
        files.extend(Path(root) / name for name in names if name.endswith(".json"))
    failed = []
    for path in tqdm(files, total=len(files)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ok, msg = validate(data, concepts)
        except Exception as exc:
            ok, msg = False, str(exc)
        if not ok:
            failed.append(f"{path}\t{msg}")
    failed_list = repo_path(args.failed_list)
    failed_list.parent.mkdir(parents=True, exist_ok=True)
    failed_list.write_text("\n".join(failed), encoding="utf-8")
    print(f"Checked {len(files)} files; failures: {len(failed)}")
    if failed:
        print(f"Failed list: {failed_list}")


if __name__ == "__main__":
    main()
