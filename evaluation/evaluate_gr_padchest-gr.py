import json
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import autocast_context, base_parser, ensure_output_dir, load_model_and_processors, repo_path, require_path, tokenize, write_json
from grounding_common import get_interpolated_map, max_point, point_in_boxes


def parse_args():
    parser = base_parser("Zero-shot grounding on PadChest-GR.")
    parser.add_argument("--json", default="data/conns_evaluation/PadChest-GR/test_grounding.json")
    parser.add_argument("--image-dir", default="data/PadChest-GR/PadChest_GR_8bit_short896")
    return parser.parse_args()


def main():
    args = parse_args()
    json_path = require_path(args.json, "PadChest-GR grounding JSON")
    image_dir = require_path(args.image_dir, "PadChest-GR image directory")
    out_dir = ensure_output_dir(args.output_dir)
    model, image_processor, tokenizer, device = load_model_and_processors(args)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = [item["text"] for item in data]
    tokenized_prompts = [tokenize(tokenizer, [prompt], max_length=128) for prompt in prompts]
    by_prompt = defaultdict(list)

    with torch.no_grad():
        for idx, item in tqdm(list(enumerate(data)), total=len(data)):
            image_path = image_dir / item["image"]
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            boxes = item["bbox"]
            if boxes and isinstance(boxes[0][0], (int, float)):
                boxes = [boxes]

            pixel_values = image_processor(image, return_tensors="pt")["pixel_values"].to(device)
            with autocast_context(device):
                outputs = model.compute_logits(pixel_values=pixel_values, encoded_key_phrases=[tokenized_prompts[idx]])
            heatmap = get_interpolated_map(outputs["similarity_scores"][0, 0], (height, width), fill_value=-1e9)
            point = max_point(heatmap)
            hit = point_in_boxes(point, boxes)
            by_prompt[item["text"]].append(hit)

    hits = [hit for prompt_hits in by_prompt.values() for hit in prompt_hits]
    if not hits:
        raise RuntimeError(f"No valid PadChest-GR grounding samples found under {repo_path(args.image_dir)}")

    print("\nPadChest-GR Grounding Results")
    mean_score = float(np.mean(hits))
    print(f"Mean pointing score: {mean_score:.4f} ({sum(hits)}/{len(hits)})")
    write_json(
        out_dir / "padchest_gr_grounding_metrics.json",
        {"mean_pointing_score": mean_score, "hits": int(sum(hits)), "total": len(hits)},
    )


if __name__ == "__main__":
    main()
