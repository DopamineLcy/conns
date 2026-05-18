import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia",
]


class ChestXray14Dataset(Dataset):
    def __init__(self, rel_paths, labels, image_root):
        self.rel_paths = rel_paths
        self.labels = labels
        self.image_root = image_root

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        image_path = self.image_root / self.rel_paths[idx]
        if not image_path.exists():
            return None, None
        try:
            return Image.open(image_path).convert("RGB"), self.labels[idx]
        except Exception:
            return None, None


def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return [], []
    images, labels = zip(*batch)
    return list(images), torch.tensor(np.stack(labels))


def parse_args():
    parser = base_parser("Zero-shot classification on NIH ChestXray14.")
    parser.add_argument("--test-list", default="data/NIH-CXR/CARZero/test_list.txt")
    parser.add_argument("--prompt-json", default="data/NIH-CXR/CARZero/chestxray14_test_text.json")
    parser.add_argument("--image-root", default="data/NIH-CXR/images")
    return parser.parse_args()


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [data[str(k)][0] for k in sorted(int(k) for k in data)]


def load_test_list(path):
    rel_paths, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1 + len(CLASSES):
                rel_paths.append(parts[0])
                labels.append([float(v) for v in parts[1 : 1 + len(CLASSES)]])
    return rel_paths, np.asarray(labels, dtype=np.float32)


def main():
    args = parse_args()
    test_list = require_path(args.test_list, "ChestXray14 test list")
    prompt_json = require_path(args.prompt_json, "ChestXray14 prompt JSON")
    image_root = require_path(args.image_root, "ChestXray14 image root")
    model, image_processor, tokenizer, device = load_model_and_processors(args)

    prompts = load_prompts(prompt_json)
    tokenized = tokenize(tokenizer, prompts, max_length=64)
    rel_paths, targets = load_test_list(test_list)
    loader = DataLoader(
        ChestXray14Dataset(rel_paths, targets, image_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    preds, valid_targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, total=len(loader)):
            if not images:
                continue
            pixel_values = image_processor(images, return_tensors="pt")["pixel_values"].to(device)
            with autocast_context(device):
                logits = model.compute_logits(pixel_values=pixel_values, encoded_key_phrases=[tokenized])["logits"]
            preds.extend(logits.sigmoid().detach().cpu().numpy())
            valid_targets.extend(labels.numpy())

    if not valid_targets:
        raise RuntimeError(f"No valid ChestXray14 images found under {repo_path(args.image_root)}")
    print_multilabel_metrics(np.asarray(valid_targets), np.asarray(preds), CLASSES, "ChestXray14 Positive Prompts")


if __name__ == "__main__":
    main()
