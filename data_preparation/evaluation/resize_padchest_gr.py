import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description="Resize PadChest-GR images to a fixed short side.")
    parser.add_argument("--input-dir", default="data/PadChest-GR/PadChest_GR_8bit")
    parser.add_argument("--output-dir", default="data/PadChest-GR/PadChest_GR_8bit_short896")
    parser.add_argument("--short-side", type=int, default=896)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = repo_path(args.input_dir)
    output_dir = repo_path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(list(input_dir.glob("*.png"))):
        img = Image.open(path)
        width, height = img.size
        if width < height:
            new_size = (args.short_side, int(height * args.short_side / width))
        else:
            new_size = (int(width * args.short_side / height), args.short_side)
        img.resize(new_size, Image.Resampling.LANCZOS).save(output_dir / path.name)


if __name__ == "__main__":
    main()
