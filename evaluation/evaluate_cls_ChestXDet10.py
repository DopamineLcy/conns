import json

import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from common import (
    autocast_context,
    base_parser,
    load_model_and_processors,
    print_multilabel_metrics,
    repo_path,
    require_path,
    tokenize,
)


CLASSES = [
    "Atelectasis",
    "Calcification",
    "Consolidation",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Fracture",
    "Mass",
    "Nodule",
    "Pneumothorax",
]


def parse_args():
    parser = base_parser("Zero-shot classification on ChestX-Det10.")
    parser.add_argument("--test-json", default="data/ChestX-Det10/test.json")
    parser.add_argument("--image-dir", default="data/ChestX-Det10/test_imgs")
    return parser.parse_args()


def main():
    args = parse_args()
    test_json = require_path(args.test_json, "ChestX-Det10 test JSON")
    image_dir = require_path(args.image_dir, "ChestX-Det10 image directory")
    model, image_processor, tokenizer, device = load_model_and_processors(args)

    prompts = [f"There is {name}." for name in CLASSES]
    tokenized = tokenize(tokenizer, prompts, max_length=128)

    with open(test_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    targets = MultiLabelBinarizer(classes=CLASSES).fit_transform([item["syms"] for item in data])

    preds, valid_targets = [], []
    with torch.no_grad():
        for idx, item in tqdm(list(enumerate(data)), total=len(data)):
            image_path = image_dir / item["file_name"]
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            pixel_values = image_processor(image, return_tensors="pt")["pixel_values"].to(device)
            with autocast_context(device):
                logits = model.compute_logits(pixel_values=pixel_values, encoded_key_phrases=[tokenized])["logits"]
            preds.append(logits.sigmoid().detach().cpu().numpy()[0])
            valid_targets.append(targets[idx])

    if not valid_targets:
        raise RuntimeError(f"No valid ChestX-Det10 images found under {repo_path(args.image_dir)}")

    valid_targets = np.asarray(valid_targets)
    print_multilabel_metrics(valid_targets, np.asarray(preds), CLASSES, "ChestX-Det10 Positive Prompts")


if __name__ == "__main__":
    main()
