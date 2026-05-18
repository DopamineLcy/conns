import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from common import autocast_context, base_parser, ensure_output_dir, load_model_and_processors, repo_path, require_path, tokenize, write_json
from grounding_common import get_interpolated_map, max_point, point_in_boxes


def parse_args():
    parser = base_parser("Zero-shot grounding on MS-CXR.")
    parser.add_argument("--csv", default="data/MS-CXR/MS_CXR_Local_Alignment_v1.1.0.csv")
    parser.add_argument("--test-json", default="data/MS-CXR/preprocess/test.json")
    parser.add_argument("--mimic-image-root", default="data/raw_dataset/MIMIC-CXR-JPG/files")
    return parser.parse_args()


def bbox_from_csv_dims(box, orig_size, csv_size):
    orig_w, orig_h = orig_size
    csv_w, csv_h = csv_size
    scale_x = orig_w / csv_w
    scale_y = orig_h / csv_h
    x1, y1, x2, y2 = box
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def filename_key(path):
    return path.split("/")[-1]


def main():
    args = parse_args()
    csv_path = require_path(args.csv, "MS-CXR CSV")
    test_json = require_path(args.test_json, "MS-CXR test JSON")
    mimic_root = require_path(args.mimic_image_root, "MIMIC-CXR-JPG files directory")
    out_dir = ensure_output_dir(args.output_dir)
    model, image_processor, tokenizer, device = load_model_and_processors(args)

    df = pd.read_csv(csv_path)
    filename_to_dims = {}
    file_prompt_to_cat = {}
    for _, row in df.iterrows():
        fname = filename_key(str(row["path"]))
        if "image_width" in row and "image_height" in row:
            filename_to_dims[fname] = (float(row["image_width"]), float(row["image_height"]))
        elif "width" in row and "height" in row:
            filename_to_dims[fname] = (float(row["width"]), float(row["height"]))
        if "label_text" in row and "category_name" in row:
            file_prompt_to_cat[(fname, str(row["label_text"]))] = str(row["category_name"])

    with open(test_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = defaultdict(list)
    with torch.no_grad():
        for item in tqdm(data, total=len(data)):
            rel_path = item.get("image") or item.get("file_name") or item.get("filename")
            if rel_path is None:
                continue
            rel_path = str(rel_path).replace("files/", "", 1)
            image_path = mimic_root / rel_path
            if not image_path.exists():
                continue

            image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = image.size
            filename = filename_key(rel_path)
            csv_dims = filename_to_dims.get(filename, (orig_w, orig_h))
            detections = item.get("det", [])
            if not detections:
                continue

            pixel_values = image_processor(image, return_tensors="pt")["pixel_values"].to(device)
            prompts = [det["name"] for det in detections]
            tokenized = tokenize(tokenizer, prompts, max_length=128)
            with autocast_context(device):
                outputs = model.compute_logits(pixel_values=pixel_values, encoded_key_phrases=[tokenized])
            sim_scores = outputs["similarity_scores"][0]

            for idx, det in enumerate(detections):
                prompt = det["name"]
                raw_boxes = det.get("label", [])
                if raw_boxes and isinstance(raw_boxes[0][0], (int, float)):
                    raw_boxes = [raw_boxes]
                boxes = [bbox_from_csv_dims(box, (orig_w, orig_h), csv_dims) for box in raw_boxes]
                if not boxes:
                    continue
                heatmap = get_interpolated_map(sim_scores[idx], (orig_h, orig_w), fill_value=-1e9)
                point = max_point(heatmap)
                hit = point_in_boxes(point, boxes)
                category = file_prompt_to_cat.get((filename, prompt), prompt)
                results[category].append(hit)

    all_hits = [hit for hits in results.values() for hit in hits]
    if not all_hits:
        raise RuntimeError(f"No valid MS-CXR grounding samples found under {repo_path(args.mimic_image_root)}")

    print("\nMS-CXR Grounding Results")
    metrics = {"per_class": {}}
    for category, hits in sorted(results.items()):
        score = float(np.mean(hits))
        metrics["per_class"][category] = {"pointing_score": score, "hits": int(sum(hits)), "total": len(hits)}
        print(f"{category:<32} pointing score: {score:.4f} ({sum(hits)}/{len(hits)})")
    print("-" * 60)
    metrics["mean_pointing_score"] = float(np.mean(all_hits))
    print(f"Mean pointing score: {metrics['mean_pointing_score']:.4f}")
    write_json(out_dir / "ms_cxr_grounding_metrics.json", metrics)


if __name__ == "__main__":
    main()
