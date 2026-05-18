import json
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import autocast_context, base_parser, ensure_output_dir, load_model_and_processors, repo_path, require_path, tokenize, write_json
from grounding_common import get_interpolated_map, max_point, point_in_boxes


FINDING_MAPPING = {
    "atelectasis": "Atelectasis",
    "tissue calcification": "Calcification",
    "pulmonary consolidation": "Consolidation",
    "pleural effusion": "Effusion",
    "pulmonary emphysema": "Emphysema",
    "fibrosis": "Fibrosis",
    "bone fracture": "Fracture",
    "pulmonary mass": "Mass",
    "lung nodule": "Nodule",
    "pneumothorax": "Pneumothorax",
}


def parse_args():
    parser = base_parser("Zero-shot grounding on ChestX-Det10.")
    parser.add_argument("--test-json", default="data/ChestX-Det10/test.json")
    parser.add_argument("--image-dir", default="data/ChestX-Det10/test_imgs")
    return parser.parse_args()


def main():
    args = parse_args()
    test_json = require_path(args.test_json, "ChestX-Det10 test JSON")
    image_dir = require_path(args.image_dir, "ChestX-Det10 image directory")
    out_dir = ensure_output_dir(args.output_dir)
    model, image_processor, tokenizer, device = load_model_and_processors(args)

    prompts = [f"There is {name}." for name in FINDING_MAPPING]
    tokenized = tokenize(tokenizer, prompts, max_length=128)

    with open(test_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = defaultdict(list)
    with torch.no_grad():
        for item in tqdm(data, total=len(data)):
            image_path = image_dir / item["file_name"]
            if not image_path.exists() or not item.get("syms"):
                continue
            sym_to_boxes = defaultdict(list)
            for sym, box in zip(item["syms"], item["boxes"]):
                sym_to_boxes[sym].append(box)

            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            pixel_values = image_processor(image, return_tensors="pt")["pixel_values"].to(device)
            with autocast_context(device):
                outputs = model.compute_logits(pixel_values=pixel_values, encoded_key_phrases=[tokenized])
            sim_scores = outputs["similarity_scores"][0]

            for prompt_idx, (finding, class_name) in enumerate(FINDING_MAPPING.items()):
                boxes = sym_to_boxes.get(class_name, [])
                if not boxes:
                    continue
                heatmap = get_interpolated_map(sim_scores[prompt_idx], (height, width))
                point = max_point(heatmap)
                hit = point_in_boxes(point, boxes)
                results[class_name].append(hit)

    all_hits = [hit for hits in results.values() for hit in hits]
    if not all_hits:
        raise RuntimeError(f"No valid ChestX-Det10 grounding samples found under {repo_path(args.image_dir)}")

    print("\nChestX-Det10 Grounding Results")
    metrics = {"per_class": {}}
    for class_name, hits in sorted(results.items()):
        score = float(np.mean(hits))
        metrics["per_class"][class_name] = {"pointing_score": score, "hits": int(sum(hits)), "total": len(hits)}
        print(f"{class_name:<24} pointing score: {score:.4f} ({sum(hits)}/{len(hits)})")
    print("-" * 60)
    metrics["mean_pointing_score"] = float(np.mean(all_hits))
    print(f"Mean pointing score: {metrics['mean_pointing_score']:.4f}")
    write_json(out_dir / "chestxdet10_grounding_metrics.json", metrics)


if __name__ == "__main__":
    main()
