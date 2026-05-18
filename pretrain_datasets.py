import json
import os
import random
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoTokenizer
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from torchvision.transforms import InterpolationMode


REPO_ROOT = Path(__file__).resolve().parent


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(REPO_ROOT / p)


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class CheXpertValDataset(Dataset):
    def __init__(self, 
                 data_root: str = "data/CheXpert",
                 csv_path: str = "data/CheXpert/val_labels.csv", 
                 image_processor = None) -> None:
        super().__init__()
        self.data_root = resolve_path(data_root)
        self.image_processor = image_processor
        csv_path = resolve_path(csv_path)
        
        # CheXpert Classes
        self.classes = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"
        ]
        
        if not os.path.exists(csv_path):
            print(f"Warning: CheXpert CSV not found at {csv_path}")
            self.df = pd.DataFrame(columns=["Path"] + self.classes)
        else:
            self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # Handle path replacement: CheXpert-v1.0/valid -> val
        img_subpath = row["Path"].replace('CheXpert-v1.0/valid', 'val')
        img_path = os.path.join(self.data_root, img_subpath)
        
        if os.path.exists(img_path):
            try:
                image = pil_loader(img_path)
                if self.image_processor:
                    processed = self.image_processor(image, return_tensors="pt")
                    pixel_values = processed['pixel_values'].squeeze(0)
                else:
                    # Fallback if no processor provided (should not happen in correct usage)
                    pixel_values = torch.zeros(3, 518, 518)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                pixel_values = torch.zeros(3, 518, 518)
        else:
            pixel_values = torch.zeros(3, 518, 518)
            
        # Get targets
        target = torch.tensor([float(row[c]) for c in self.classes], dtype=torch.float)
        
        return {
            "pixel_values": pixel_values,
            "target": target,
            "path": img_subpath
        }

class MIMICConnsDataset(Dataset):
    def __init__(self, data_root: str, is_train: bool = True, args: Any = None) -> None:
        super().__init__()
        self.data_root = data_root
        self.is_train = is_train
        self.args = args
        self.image_root = resolve_path(self.data_root)
        self.report_root = resolve_path(getattr(args, "report_root", "data/conns_training/reports_extract_concepts"))

        if args.all_view:
            meta_csv = getattr(args, "metadata_csv", "data/conns_training/mimic_conns_training.csv")
        else:
            meta_csv = getattr(args, "metadata_csv_frontal", "data/conns_training/mimic_conns_training_frontal.csv")
        meta_csv = resolve_path(meta_csv)
        df = pd.read_csv(meta_csv).astype({'dicom_id': str, 'study_id': str, 'subject_id': str, 'split': str})

        target_split = 'train' if self.is_train else 'validate'
        self.df = df[df['split'] == target_split].reset_index(drop=True)

        concepts_path = resolve_path(getattr(args, "concepts_path", "data/conns_training/concepts.json"))
        with open(concepts_path, "r", encoding="utf-8") as f:
            self.cols: List[str] = json.load(f)

        self.num_entities = len(self.cols)
        
        self.dicom_ids = self.df["dicom_id"].astype(str).values
        self.subject_ids = self.df["subject_id"].astype(int).values
        self.study_ids = self.df["study_id"].astype(int).values

        model_name = resolve_path(getattr(args, "vision_model_path", "external/rad-dino-maira-2"))
        self.rad_dino_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        text_model_path = resolve_path(getattr(args, "text_model_path", "external/BiomedVLP-CXR-BERT-specialized"))
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_path, trust_remote_code=True)

        self.presence_map = {"Yes": 1, "No": 0, "Not Mentioned": 0, "Uncertain": -100}
        
        self.special_classes = {
            "normal lung transparency",
            "normal cardiac silhouette", 
            "normal hilar contour",
            "intact osseous structures"
        }

        print("num_entities: ", self.num_entities)
        print("special_classes: ", self.special_classes)
        self.special_class_sampling_prob = getattr(args, 'special_class_sampling_prob', 0.5) if args else 0.5
        
        self.entity_sampling_counts = defaultdict(int)  # {entity_name: count}
        self.total_batches_processed = 0

        yes_expressions_dir = resolve_path(getattr(args, "yes_expressions_dir", "data/conns_training/yes_expressions"))
        no_expressions_dir = resolve_path(getattr(args, "no_expressions_dir", "data/conns_training/no_expressions"))
        
        self.yes_fillings = {}  # {label: segments_list}
        self.no_fillings = {}  # {label: segments_list}
        
        for label in self.cols:
            # --- Load Yes Expressions ---
            filename = label.replace(' ', '_') + '_segments_train.json'
            file_path = os.path.join(yes_expressions_dir, filename)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    segment_counts = []
                    if "segments_by_frequency" in data:
                        for item in data["segments_by_frequency"]:
                            segment = item["segment"]
                            count = int(item.get("count", 1))
                            segment_counts.append((segment, count))
                    
                    if segment_counts:
                        segment_counts.sort(key=lambda x: x[1], reverse=True)
                        top_segments = [s for s, c in segment_counts[:10]]
                        self.yes_fillings[label] = top_segments
                    else:
                        self.yes_fillings[label] = []
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
                    self.yes_fillings[label] = []
            else:
                self.yes_fillings[label] = []

            # --- Load No Expressions ---
            file_path_no = os.path.join(no_expressions_dir, filename)
            
            if os.path.exists(file_path_no):
                try:
                    with open(file_path_no, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    segment_counts = []
                    if "segments_by_frequency" in data:
                        for item in data["segments_by_frequency"]:
                            segment = item["segment"]
                            count = int(item.get("count", 1))
                            segment_counts.append((segment, count))
                    
                    if segment_counts:
                        segment_counts.sort(key=lambda x: x[1], reverse=True)
                        top_segments = [s for s, c in segment_counts[:10]]
                        self.no_fillings[label] = top_segments
                    else:
                        self.no_fillings[label] = []
                except Exception as e:
                    print(f"Warning: Failed to load {file_path_no}: {e}")
                    self.no_fillings[label] = []
            else:
                self.no_fillings[label] = []

        if self.is_train:
            # Augmentation parameters from args or default
            degrees = getattr(self.args, 'aug_degrees', 20)
            shear = getattr(self.args, 'aug_shear', 10)
            translate = getattr(self.args, 'aug_translate', (0.1, 0.1))
            scale = getattr(self.args, 'aug_scale', (0.9, 1.0))

            self.transform = transforms.Compose([
                # transforms.CenterCrop((518, 518)),
                # transforms.RandomAffine(degrees=degrees, shear=shear, translate=translate, scale=scale),
                transforms.RandomResizedCrop((518, 518), scale=scale, ratio=(0.9, 1.1), interpolation=InterpolationMode.BICUBIC),

                transforms.RandomApply([
                    transforms.RandomAffine(degrees=degrees, interpolation=InterpolationMode.BICUBIC),
                ], p=self.args.aug_prob),

                transforms.RandomApply([
                    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
                ], p=self.args.aug_prob),
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.df)
    
    def _weighted_random_choice(self, label):
        """
        Randomly select one of the cached top positive segments.
        
        Args:
            label: Concept name used to look up cached segments.
        
        Returns:
            A segment string, or None when no segment is available.
        """
        segments = self.yes_fillings.get(label, [])
        
        if not segments:
            return None
        
        return random.choice(segments)
    
    def _weighted_random_choice_neg(self, label):
        """
        Randomly select one of the cached top negative segments.
        
        Args:
            label: Concept name used to look up cached segments.
        
        Returns:
            A segment string, or None when no segment is available.
        """
        segments = self.no_fillings.get(label, [])
        
        if not segments:
            return None
        
        return random.choice(segments)

    def _build_paths(self, dicom_id: str, subject_id: int, study_id: int) -> Tuple[str, str]:
        sid_str = str(subject_id)
        stid_str = str(study_id)
        prefix = sid_str[:2]
        rel_dir = os.path.join(f"p{prefix}", f"p{sid_str}", f"s{stid_str}")
        rel_dir_no_study = os.path.join(f"p{prefix}", f"p{sid_str}")
        image_path = os.path.join(self.image_root, rel_dir, f"{dicom_id}.jpg")
        report_path = os.path.join(self.report_root, rel_dir_no_study, f"s{stid_str}.json")
        return image_path, report_path

    def _load_report(self, report_path: str) -> Dict[str, Any]:
        if not os.path.exists(report_path): return {}
        with open(report_path, "r", encoding="utf-8") as f: return json.load(f)

    def yes_template(self, entity_name: str) -> str:
        """
        Return the positive template for an entity.
        """
        # if random.random() < 0.5:
        return f"There is {entity_name}."
        # return f"{entity_name}"

    def no_template(self, entity_name: str) -> str:
        """
        Return the negative template for an entity.
        """
        # if random.random() < 0.5:
        return f"There is no {entity_name}."
        # return f"No {entity_name}"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dicom_id = self.dicom_ids[idx]
        subject_id = self.subject_ids[idx]
        study_id = self.study_ids[idx]

        image_path, report_path = self._build_paths(dicom_id, subject_id, study_id)
        
        if os.path.exists(image_path):
            image = pil_loader(image_path)
            if self.is_train and self.transform is not None and self.args.is_augmentation:
                image = self.transform(image)
                
            processed = self.rad_dino_processor(image, return_tensors="pt")
            processed_image = processed['pixel_values'].squeeze(0)
        else:
            raise FileNotFoundError(f"Image not found: {image_path}")

        report = self._load_report(report_path)

        return {
            "image": processed_image,
            "report_path": report_path,
            "report_dict": report
        }

    def _get_attribute_str(self, info: Dict[str, Any]) -> str:
        """
        Extract and combine location and characteristics into a single string.
        """
        loc = info.get("location", "")
        if loc is None: loc = ""
        if str(loc) == "NA": loc = "uncertain"
        
        attrs = info.get("characteristics", [])
        if attrs is None: attrs = []
        if isinstance(attrs, str): attrs = [attrs]
        
        # Filter out None or empty strings from attrs
        attrs = [str(a) for a in attrs if a]
        
        parts = []
        if loc: parts.append(f"location is {loc}.")
        if attrs: parts.append(f"Attributes are {', '.join(attrs)}.")
        
        return " ".join(parts).strip()

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.stack([b["image"] for b in batch], dim=0)
        
        report_paths_batch = []
        
        # We need to collect flattened lists
        flat_report_texts = []
        flat_text_is_pos = []
        flat_sampled_entity_ids = []
        flat_src_indices = []
        flat_text_attributes = [] # [Total_Text_Samples]
        
        presence_values_list = []
        batch_image_attributes_list = [] # [B, 37] strings
        
        batch_sampling_counts = defaultdict(int)
        
        K = 4

        for b_idx, b in enumerate(batch):
            report_path = b["report_path"]
            report_paths_batch.append(report_path)
            report_dict = b["report_dict"]
            current_presence_values = []
            current_image_attributes = [] # [37]
            pos_indices = []
            neg_indices = []
            
            # --- 1. Build the full mask. ---
            for idx, entity_name in enumerate(self.cols):
                info = report_dict.get(entity_name, {})
                presence_str = info.get("presence", "error")
                assert presence_str != "error", f"Error in report_dict: {report_dict}"

                # Extract attributes for this entity in this image
                attr_str = self._get_attribute_str(info)
                current_image_attributes.append(attr_str)

                if presence_str == "Not Mentioned" and entity_name in self.special_classes:
                    val = -100
                else:
                    val = self.presence_map[presence_str]
                
                current_presence_values.append(val)

                # Split valid entities for sampling.
                if val == 1:
                    pos_indices.append(idx)
                elif val == 0:
                    neg_indices.append(idx)

            presence_values_list.append(torch.tensor(current_presence_values, dtype=torch.long))
            batch_image_attributes_list.append(current_image_attributes)

            # --- 2. Sampling. ---
            current_sampled_indices = []

            special_pos_indices = []
            normal_pos_indices = []
            for idx in pos_indices:
                entity_name = self.cols[idx]
                if entity_name in self.special_classes:
                    if random.random() < self.special_class_sampling_prob:
                        special_pos_indices.append(idx)
                else:
                    normal_pos_indices.append(idx)

            random.shuffle(normal_pos_indices)
            random.shuffle(special_pos_indices)
            
            sampled_normal = normal_pos_indices[:K]
            remaining_slots = K - len(sampled_normal)
            sampled_special = special_pos_indices[:remaining_slots] if remaining_slots > 0 else []
            
            current_sampled_indices.extend(sampled_normal)
            current_sampled_indices.extend(sampled_special)
            
            actual_K_pos = len(current_sampled_indices)
            actual_K_neg =  K - actual_K_pos
            # print("actual_K_pos: ", actual_K_pos, "actual_K_neg: ", actual_K_neg)
            random.shuffle(neg_indices)
            current_sampled_indices.extend(neg_indices[:actual_K_neg])

            assert len(current_sampled_indices) == K, f"len(current_sampled_indices) != K: {len(current_sampled_indices)} != {K}"
        
            # --- 3. Build and flatten text samples. ---
            for entity_idx in current_sampled_indices:
                entity_name = self.cols[entity_idx]
                
                self.entity_sampling_counts[entity_name] += 1
                batch_sampling_counts[entity_name] += 1
                
                is_presence_yes = current_presence_values[entity_idx]
                
                info = report_dict.get(entity_name, {})
                presence_str = info.get("presence", "error")

                # 50% chance to use template text, 50% to use relevant segment
                if random.random() < self.args.there_is_prob and entity_name not in self.special_classes:
                    current_image_entity_attributes = "location is uncertain."
                    if is_presence_yes == 0:
                        text_content = self.no_template(entity_name)
                    else:
                        text_content = self.yes_template(entity_name)
                else:
                    current_image_entity_attributes = current_image_attributes[entity_idx]
                    if presence_str == "Not Mentioned":
                        text_content = self._weighted_random_choice_neg(entity_name)
                    else:
                        raw_analysis = info.get("evidential_segment", "error")
                        assert raw_analysis != "error", f"Error in report_dict: {report_dict}"
                        text_content = raw_analysis
                    
                    if text_content is None or len(text_content)<2:
                        if is_presence_yes == 0:
                            text_content = self.no_template(entity_name)
                        else:
                            text_content = self.yes_template(entity_name)

                flat_report_texts.append(text_content)
                flat_text_is_pos.append(is_presence_yes)
                flat_sampled_entity_ids.append(entity_idx)
                flat_src_indices.append(b_idx)
                
                # Get attribute string for this text sample (derived from the source image's entity info)
                # Note: We use the SAME attributes as the image it came from
                flat_text_attributes.append(current_image_entity_attributes)

                if getattr(self.args, "use_counterfactual", False):
                    use_mining = random.random() < 0.5
                    if is_presence_yes == 0:
                        # Real is No -> CF is Yes (text_is_pos == 1)
                        cf_text = None
                        if use_mining:
                            cf_text = self._weighted_random_choice(entity_name)
                        
                        if cf_text is None or len(cf_text)<2:
                            cf_text = self.yes_template(entity_name)
                        cf_is_pos = 1
                        flat_report_texts.append(cf_text)
                        flat_text_is_pos.append(cf_is_pos)
                        flat_sampled_entity_ids.append(entity_idx)
                        flat_src_indices.append(b_idx)
                        # Counterfactual: Attributes might not match the text description anymore
                        # But for "hard negative" mining, we usually care about the Positive Text vs Negative Image logic
                        # or Positive Text vs Positive Image (Different Instance).
                        # If this text is CF (fabricated), it's "fake".
                        # However, to be safe, we can pass empty string or the original attributes.
                        # User logic is about "same entity", "contradiction".
                        # If this is a CF text saying "There is Edema" (when original was No), 
                        # does it have "left mild" attributes? No.
                        # So we should probably pass empty string for CF texts to avoid mining hard negatives based on fake attributes.
                        flat_text_attributes.append("")

                    elif is_presence_yes == 1:
                        cf_text = None
                        if use_mining:
                            cf_text = self._weighted_random_choice_neg(entity_name)
                        
                        if cf_text is None or len(cf_text)<2:
                            cf_text = self.no_template(entity_name)
                        cf_is_pos = 0
                        flat_report_texts.append(cf_text)
                        flat_text_is_pos.append(cf_is_pos)
                        flat_sampled_entity_ids.append(entity_idx)
                        flat_src_indices.append(b_idx)
                        flat_text_attributes.append("") # CF text has no real attributes

        presence_values = torch.stack(presence_values_list, dim=0)
        
        # Tokenize (CPU side)
        tokenized = self.tokenizer(
            flat_report_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        )

        self.total_batches_processed += 1

        return {
            "images": images,
            "report_paths": report_paths_batch,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "text_is_pos": torch.tensor(flat_text_is_pos, dtype=torch.long),
            "text_entity_ids": torch.tensor(flat_sampled_entity_ids, dtype=torch.long),
            "text_src_indices": torch.tensor(flat_src_indices, dtype=torch.long),
            "text_attributes": flat_text_attributes, # List[str] length N
            "image_attributes": batch_image_attributes_list, # List[List[str]] [B, 37]
            "presence_values": presence_values,
            "_sampling_counts": dict(batch_sampling_counts)
        }
    
    def plot_entity_sampling_histogram(self, save_path=None, figsize=(16, 8)):
        """
        Generate and optionally save the entity sampling histogram.
        
        Args:
            save_path: Output path. If None, the figure is not saved.
            figsize: Figure size.
        """
        if not self.entity_sampling_counts:
            print("Warning: No entity sampling data available.")
            return None
        
        entities = list(self.entity_sampling_counts.keys())
        counts = [self.entity_sampling_counts[entity] for entity in entities]
        
        sorted_data = sorted(zip(entities, counts), key=lambda x: x[1], reverse=True)
        sorted_entities, sorted_counts = zip(*sorted_data)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(sorted_entities)), sorted_counts, align='center')
        
        ax.set_yticks(range(len(sorted_entities)))
        ax.set_yticklabels(sorted_entities, fontsize=8)
        ax.invert_yaxis()
        
        ax.set_xlabel('Sampling Count', fontsize=12)
        ax.set_ylabel('Entity Name', fontsize=12)
        ax.set_title(f'Entity Sampling Frequency (Total Batches: {self.total_batches_processed})', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, count) in enumerate(zip(bars, sorted_counts)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {count}', ha='left', va='center', fontsize=7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Entity sampling histogram saved to: {save_path}")
        
        return fig
    
    def get_sampling_statistics(self):
        """
        Return entity sampling statistics.
        
        Returns:
            A dictionary with sampling statistics.
        """
        if not self.entity_sampling_counts:
            return {
                'total_entities': len(self.cols),
                'sampled_entities': 0,
                'total_samples': 0,
                'avg_samples_per_entity': 0,
                'max_samples': 0,
                'min_samples': 0,
                'total_batches': self.total_batches_processed,
                'entity_counts': {}
            }
        
        counts = list(self.entity_sampling_counts.values())
        return {
            'total_entities': len(self.cols),
            'sampled_entities': len(self.entity_sampling_counts),
            'total_samples': sum(counts),
            'avg_samples_per_entity': np.mean(counts) if counts else 0,
            'max_samples': max(counts) if counts else 0,
            'min_samples': min(counts) if counts else 0,
            'total_batches': self.total_batches_processed,
            'entity_counts': dict(self.entity_sampling_counts)
        }


class MIMICConnsClassificationDataset(MIMICConnsDataset):
    def __init__(self, data_root: str, is_train: bool = False, args: Any = None) -> None:
        super().__init__(data_root, is_train=is_train, args=args)
        self.is_train = False
        self.transform = None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dicom_id = self.dicom_ids[idx]
        subject_id = self.subject_ids[idx]
        study_id = self.study_ids[idx]

        image_path, report_path = self._build_paths(dicom_id, subject_id, study_id)
        
        if os.path.exists(image_path):
            image = pil_loader(image_path)
            processed = self.rad_dino_processor(image, return_tensors="pt")
            processed_image = processed['pixel_values'].squeeze(0)
        else:
            processed_image = torch.zeros(3, 518, 518)

        report = self._load_report(report_path)
        
        gt_presence_list = []
        pos_texts = []
        neg_texts = []

        for entity_name in self.cols:
            info = report.get(entity_name, {})
            presence_str = info.get("presence", "Not Mentioned") # Default to Not Mentioned if missing
            
            # Map presence
            if presence_str == "Yes":
                val = 1
                pos_text = f"There is {entity_name}."
            elif presence_str == "No" or presence_str == "Not Mentioned":
                val = 0
                pos_text = f"There is {entity_name}."
            else: # Uncertain or Error
                val = -100
                pos_text = f"There is {entity_name}." # Placeholder

            neg_text = f"There is no {entity_name}."
            
            gt_presence_list.append(val)
            pos_texts.append(pos_text)
            neg_texts.append(neg_text)

        return {
            "image": processed_image,
            "gt_presence": torch.tensor(gt_presence_list, dtype=torch.long),
            "pos_texts": pos_texts,
            "neg_texts": neg_texts
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.stack([b["image"] for b in batch], dim=0)
        gt_presence = torch.stack([b["gt_presence"] for b in batch], dim=0)
        
        # Flatten texts for tokenization: [B, 37] -> [B*37]
        flat_pos_texts = []
        flat_neg_texts = []
        for b in batch:
            flat_pos_texts.extend(b["pos_texts"])
            flat_neg_texts.extend(b["neg_texts"])
            
        tokenized_pos = self.tokenizer(
            flat_pos_texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        )
        tokenized_neg = self.tokenizer(
            flat_neg_texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        )
        
        return {
            "images": images,
            "gt_presence": gt_presence,
            "pos_input_ids": tokenized_pos["input_ids"],
            "pos_attention_mask": tokenized_pos["attention_mask"],
            "neg_input_ids": tokenized_neg["input_ids"],
            "neg_attention_mask": tokenized_neg["attention_mask"],
            "is_classification_val": True # Flag for evaluate function
        }

MIMICConns = MIMICConnsDataset
globals()["MIMIC-CoNNS"] = MIMICConnsDataset
globals()["MIMIC-CoNNS-Classification"] = MIMICConnsClassificationDataset
