import csv

import numpy as np
import torch
from PIL import Image
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


CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


def parse_args():
    parser = base_parser("Zero-shot classification on CheXpert.")
    parser.add_argument("--test-csv", default="data/CheXpert/test_labels.csv")
    parser.add_argument("--image-root", default="data/CheXpert")
    return parser.parse_args()


def resolve_chexpert_path(root, rel_path):
    if "valid" in rel_path:
        rel_path = rel_path.replace("CheXpert-v1.0/valid", "val")
    return root / rel_path


def main():
    args = parse_args()
    test_csv = require_path(args.test_csv, "CheXpert test CSV")
    image_root = require_path(args.image_root, "CheXpert image root")
    model, image_processor, tokenizer, device = load_model_and_processors(args)

    prompts = list(CLASSES)
    tokenized = tokenize(tokenizer, prompts, max_length=64)

    rows = []
    with open(test_csv, "r", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))

    preds, targets = [], []
    with torch.no_grad():
        for row in tqdm(rows, total=len(rows)):
            image_path = resolve_chexpert_path(image_root, row["Path"])
            if not image_path.exists():
                continue
            target = [float(row[name]) for name in CLASSES]
            image = Image.open(image_path).convert("RGB")
            pixel_values = image_processor(image, return_tensors="pt")["pixel_values"].to(device)
            with autocast_context(device):
                logits = model.compute_logits(pixel_values=pixel_values, encoded_key_phrases=[tokenized])["logits"]
            preds.append(logits.sigmoid().detach().cpu().numpy()[0])
            targets.append(target)

    if not targets:
        raise RuntimeError(f"No valid CheXpert images found under {repo_path(args.image_root)}")

    targets = np.asarray(targets)
    print_multilabel_metrics(targets, np.asarray(preds), CLASSES, "CheXpert Positive Prompts")


if __name__ == "__main__":
    main()
