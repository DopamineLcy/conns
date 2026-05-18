import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (
    autocast_context,
    base_parser,
    ensure_output_dir,
    load_model_and_processors,
    print_multilabel_metrics,
    repo_path,
    require_path,
    tokenize,
)


class PadChestGRDataset(Dataset):
    def __init__(self, image_ids, image_dir):
        self.image_ids = list(image_ids)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        path = self.image_dir / self.image_ids[idx]
        if not path.exists():
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
    parser = base_parser("Zero-shot classification on PadChest-GR.")
    parser.add_argument("--csv", default="data/conns_evaluation/PadChest-GR/test_binary_classification.csv")
    parser.add_argument("--image-dir", default="data/PadChest-GR/PadChest_GR_8bit_short896")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = require_path(args.csv, "PadChest-GR classification CSV")
    image_dir = require_path(args.image_dir, "PadChest-GR image directory")
    out_dir = ensure_output_dir(args.output_dir)
    model, image_processor, tokenizer, device = load_model_and_processors(args)

    df = pd.read_csv(csv_path)
    class_names = [col for col in df.columns if col != "ImageID"]
    prompts = [f"There is {name}." for name in class_names]
    tokenized = tokenize(tokenizer, prompts, max_length=128)

    labels = df[class_names].values.astype(np.float32)
    loader = DataLoader(
        PadChestGRDataset(df["ImageID"].tolist(), image_dir),
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
        raise RuntimeError(f"No valid PadChest-GR images found under {repo_path(args.image_dir)}")

    targets = np.asarray(valid_targets)
    preds = np.asarray(preds)
    print_multilabel_metrics(targets, preds, class_names, "PadChest-GR Positive Prompts")
    pd.DataFrame({"class": class_names}).to_csv(out_dir / "padchest_gr_classes.csv", index=False)


if __name__ == "__main__":
    main()
