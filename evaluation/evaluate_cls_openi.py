import json

import numpy as np
import pandas as pd
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


PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "Fracture",
    "Opacity",
    "Lesion",
    "Calcified Granuloma",
    "Granuloma",
    "No_Finding",
]

MAPPING = {
    "Pleural_Thickening": ["pleural thickening"],
    "Infiltration": ["infiltrate"],
    "Atelectasis": ["atelectases"],
}


class OpenIDataset(Dataset):
    def __init__(self, file_names, image_dir):
        self.file_names = file_names
        self.image_dir = image_dir

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = resolve_image_path(self.file_names[idx], self.image_dir)
        if path is None:
            return None, idx
        try:
            return Image.open(path).convert("RGB"), idx
        except Exception:
            return None, idx


def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return [], []
    images, indices = zip(*batch)
    return list(images), list(indices)


def parse_args():
    parser = base_parser("Zero-shot classification on Open-I.")
    parser.add_argument("--label-csv", default="data/Open-I/CARZero/custom.csv")
    parser.add_argument("--prompt-json", default="data/Open-I/CARZero/openi_multi_label_text.json")
    parser.add_argument("--image-dir", default="data/Open-I/images/images_normalized")
    return parser.parse_args()


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [data[str(k)][0] for k in sorted(int(k) for k in data)]


def resolve_image_path(file_name, image_dir):
    candidates = [image_dir / file_name]
    if file_name.endswith(".png") and not file_name.endswith(".dcm.png"):
        candidates.append(image_dir / file_name.replace(".png", ".dcm.png"))
    if file_name.startswith("CXR"):
        trimmed = file_name[3:]
        candidates.append(image_dir / trimmed)
        if trimmed.endswith(".png") and not trimmed.endswith(".dcm.png"):
            candidates.append(image_dir / trimmed.replace(".png", ".dcm.png"))
    for path in candidates:
        if path.exists():
            return path
    return None


def build_labels(label_csv):
    df = pd.read_csv(label_csv)
    labels_text = df["labels_automatic"].fillna("")
    gt = []
    for pathology in PATHOLOGIES:
        mask = labels_text.str.contains(pathology.lower(), case=False, na=False)
        for synonym in MAPPING.get(pathology, []):
            mask |= labels_text.str.contains(synonym.lower(), case=False, na=False)
        gt.append(mask.values.astype(np.float32))
    gt = np.asarray(gt).T
    positive_sum = gt[:, :-1].sum(axis=1)
    gt[positive_sum == 0, -1] = 1.0
    return df["file_name"].tolist(), gt[:, :-1]


def display_names():
    names = PATHOLOGIES[:-1]
    names = [n.replace("Opacity", "Lung Opacity") for n in names]
    names = [n.replace("Lesion", "Lung Lesion") for n in names]
    names = [n.replace("Pleural_Thickening", "Pleural Thickening") for n in names]
    names = [n.replace("Infiltration", "Infiltrate") for n in names]
    names = [n.replace("Atelectasis", "Atelectases") for n in names]
    return names


def main():
    args = parse_args()
    label_csv = require_path(args.label_csv, "Open-I label CSV")
    prompt_json = require_path(args.prompt_json, "Open-I prompt JSON")
    image_dir = require_path(args.image_dir, "Open-I image directory")
    model, image_processor, tokenizer, device = load_model_and_processors(args)

    tokenized = tokenize(tokenizer, load_prompts(prompt_json), max_length=64)
    file_names, labels = build_labels(label_csv)
    loader = DataLoader(
        OpenIDataset(file_names, image_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    preds, valid_targets = [], []
    with torch.no_grad():
        for images, indices in tqdm(loader, total=len(loader)):
            if not images:
                continue
            pixel_values = image_processor(images, return_tensors="pt")["pixel_values"].to(device)
            with autocast_context(device):
                logits = model.compute_logits(pixel_values=pixel_values, encoded_key_phrases=[tokenized])["logits"]
            preds.extend(logits.sigmoid().detach().cpu().numpy())
            valid_targets.extend(labels[list(indices)])

    if not valid_targets:
        raise RuntimeError(f"No valid Open-I images found under {repo_path(args.image_dir)}")
    print_multilabel_metrics(np.asarray(valid_targets), np.asarray(preds), display_names(), "Open-I Positive Prompts")


if __name__ == "__main__":
    main()
