import argparse
import concurrent.futures
import os
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description="Extract entity JSON from MIMIC-CXR reports with an OpenAI-compatible LLM server.")
    parser.add_argument("--prompt-file", default="data_preparation/training/prompt.txt")
    parser.add_argument("--input-dir", default="data/raw_dataset/MIMIC-CXR-JPG/reports")
    parser.add_argument("--output-dir", default="data/conns_training/reports_extract_concepts")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--model", default="./Llama-3.3-70B-Instruct-NVFP4")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--beginning-of-subject-id", default=None, help="Optional shard such as p10.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_pairs(input_dir, output_dir):
    pairs = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if not name.lower().endswith(".txt"):
                continue
            src = Path(root) / name
            rel = src.relative_to(input_dir).with_suffix(".json")
            pairs.append((src, output_dir / rel))
    return pairs


def shard_filter(pair, prefix):
    if not prefix:
        return True
    parts = pair[0].parts
    return len(parts) >= 3 and parts[-3].startswith(prefix)


def extract_one(pair, prompt, args):
    from openai import OpenAI

    src, dst = pair
    if args.skip_existing and dst.exists():
        return str(dst), "skipped"
    report_text = src.read_text(encoding="utf-8")
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": "You are an experienced radiologist. Identify information of entities in radiology reports.",
            },
            {"role": "user", "content": f"{prompt}\n\nReport Content:\n{report_text}"},
        ],
        max_tokens=4096,
        temperature=0.0,
        stream=False,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(response.choices[0].message.content, encoding="utf-8")
    return str(dst), "ok"


def main():
    args = parse_args()
    prompt = repo_path(args.prompt_file).read_text(encoding="utf-8")
    input_dir = repo_path(args.input_dir)
    output_dir = repo_path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"report directory not found: {input_dir}")
    pairs = [p for p in build_pairs(input_dir, output_dir) if shard_filter(p, args.beginning_of_subject_id)]
    print(f"Processing {len(pairs)} reports")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_one, pair, prompt, args) for pair in pairs]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            path, status = future.result()
            if status != "ok":
                tqdm.write(f"{status}: {path}")


if __name__ == "__main__":
    main()
