import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder
from conns.loss import ConnsLoss

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(REPO_ROOT / p)


def use_cuda_backend(args=None):
    device = str(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    return device.startswith("cuda") and torch.cuda.is_available()


def attention_backend(args=None):
    return "flash_attention_2" if use_cuda_backend(args) else "sdpa"


def encoder_dtype(args=None):
    return torch.bfloat16 if use_cuda_backend(args) else torch.float32


class VisionEncoder(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        model_name = resolve_path(getattr(args, "vision_model_path", "external/rad-dino-maira-2"))
        self.compute_dtype = encoder_dtype(args)
        self.rad_dino_model = AutoModel.from_pretrained(model_name, attn_implementation=attention_backend(args), dtype=self.compute_dtype)
        for param in self.rad_dino_model.parameters(): param.requires_grad = False
        self.rad_dino_model.eval()
        self.rad_dino_output_layer = getattr(args, "rad_dino_output_layer", -1)
        if self.rad_dino_output_layer != -1:
            self.layer_norm = nn.LayerNorm(768)
        self.feature_dim = 768

        self.use_extra_pos_embed = getattr(args, "use_extra_pos_embed", False)
        if self.use_extra_pos_embed:
            num_patches = 1369 + 1 
            self.extra_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.feature_dim))
            nn.init.trunc_normal_(self.extra_pos_embed, std=0.02)

        dinov2_config = Dinov2Config(hidden_size=768, num_hidden_layers=args.num_hidden_layers if args else 2, use_layer_norm=False, attn_implementation=attention_backend(args))
        print(f"Using DINOv2 with {args.num_hidden_layers} hidden layers")
        self.transformer_blocks = Dinov2Encoder(dinov2_config)

    def forward(self, images):
        device = images.device
        with torch.no_grad():
            if images.dtype != self.compute_dtype:
                images = images.to(self.compute_dtype)
            inputs = {'pixel_values': images.to(device)}
            if self.rad_dino_output_layer == -1:
                outputs = self.rad_dino_model(**inputs)
                patch_features = outputs.last_hidden_state
            else:
                outputs = self.rad_dino_model(**inputs, output_hidden_states=True)
                patch_features = outputs.hidden_states[self.rad_dino_output_layer]
                patch_features = self.layer_norm(patch_features)
        
        if self.use_extra_pos_embed:
            patch_features = patch_features + self.extra_pos_embed
        transformer_dtype = next(self.transformer_blocks.parameters()).dtype
        patch_features = patch_features.to(transformer_dtype)
        outputs = self.transformer_blocks(patch_features)
        return outputs["last_hidden_state"]

class TextEncoder(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        url = resolve_path(getattr(args, "text_model_path", "external/BiomedVLP-CXR-BERT-specialized"))
        self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(url, trust_remote_code=True, attn_implementation="sdpa", dtype=encoder_dtype(args))
        for param in self.model.parameters(): param.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None):
        if input_ids.dim() == 3:
            batch_size, num_entities, seq_len = input_ids.shape
        elif input_ids.dim() == 2:
            batch_size, seq_len = input_ids.shape
            num_entities = 1
            input_ids = input_ids.unsqueeze(1)
            if attention_mask is not None and attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1)
        else:
             batch_size, num_entities, seq_len = input_ids.shape

        outputs = self.model(input_ids=input_ids.view(-1, seq_len), attention_mask=attention_mask.view(-1, seq_len), return_dict=True)
        return outputs.last_hidden_state[:, 0, :].view(batch_size, num_entities, -1)

class CoNNSModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.vision_encoder = VisionEncoder(args=args)
        self.text_encoder = TextEncoder(args=args)
        
        proj_dim = args.proj_dim if args else None
        print(f"Using projection dimension: {proj_dim}")
        self.vision_proj = nn.Linear(self.vision_encoder.feature_dim, proj_dim, bias=False)
        self.text_proj = nn.Linear(768, proj_dim, bias=False)
        self._init_weights(self.vision_proj); self._init_weights(self.text_proj)
        self.vision_encoder.transformer_blocks.apply(self._init_weights)

        print(f"Using vision cls token: {args.use_vision_cls_token if args else False}")
        self.criterion = ConnsLoss(
            hidden_dim=proj_dim,
            attn_temperature=args.attn_temperature if args else None,
            use_vision_cls_token=args.use_vision_cls_token if args else False,
            init_logit_scale=args.init_logit_scale if args else np.log(10),
            init_logit_bias=args.init_logit_bias if args else -5.0,
            sim_op="cos", 
            world_size=self.args.world_size,
            rank=self.args.rank,
            nli_model_path=getattr(args, "nli_model_path", "cross-encoder/nli-deberta-v3-small"),
            load_nli_model=getattr(args, "load_nli_model", True),
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
    
    def forward(self, batch, device, debug=True):
        images = batch['images'].to(device, non_blocking=True)
        # report_paths = batch['report_paths']
        presence_values = batch['presence_values'].to(device, non_blocking=True) 
        
        # Pre-computed flattened inputs from collate_fn
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attn_mask = batch['attention_mask'].to(device, non_blocking=True)
        text_is_pos_tensor = batch['text_is_pos'].to(device, non_blocking=True)
        sampled_entity_ids_tensor = batch['text_entity_ids'].to(device, non_blocking=True)
        src_indices_tensor = batch['text_src_indices'].to(device, non_blocking=True)
        text_attributes = batch.get('text_attributes', None)
        image_attributes = batch.get('image_attributes', None)
        
        # Unwrap if wrapped (to avoid pin_memory overhead)
        if hasattr(text_attributes, 'data'): text_attributes = text_attributes.data
        if hasattr(image_attributes, 'data'): image_attributes = image_attributes.data

        B_total = input_ids.shape[0]
        input_ids = input_ids.view(B_total, 1, -1)
        attn_mask = attn_mask.view(B_total, 1, -1)

        text_feats = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        text_feats = self.text_proj(text_feats) 
        text_features_flat = text_feats.view(-1, self.args.proj_dim)

        image_tokens = self.vision_encoder(images)
        image_tokens = self.vision_proj(image_tokens)

        loss_output = self.criterion(
            vision_tokens=image_tokens,
            text_features=text_features_flat,
            image_presence_map=presence_values,  # [N_total]
            text_entity_ids=sampled_entity_ids_tensor, # [N_total]
            text_is_pos=text_is_pos_tensor,    # [N_total]
            text_src_indices=src_indices_tensor,
            text_attributes=text_attributes,
            image_attributes=image_attributes
        )

        return loss_output

    def compute_logits(
        self,
        pixel_values,
        encoded_key_phrases,
        **kwargs,
    ):
        # 1. Vision Forward
        image_tokens = self.vision_encoder(pixel_values)
        image_tokens = self.vision_proj(image_tokens)

        # 2. Text Forward
        # encoded_key_phrases is expected to be a list containing the batch encoding
        if isinstance(encoded_key_phrases, list):
            batch_encoding = encoded_key_phrases[0]
        else:
            batch_encoding = encoded_key_phrases

        device = pixel_values.device
        input_ids = batch_encoding["input_ids"].to(device)
        attention_mask = batch_encoding["attention_mask"].to(device)

        # TextEncoder expects [Batch, Num_Entities, Seq_Len]
        # Treat input [N, Seq_Len] as [1, N, Seq_Len]
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(0) 
            attention_mask = attention_mask.unsqueeze(0)
        elif input_ids.dim() == 3:
            # Already [B, N, L]
            pass
        else:
             raise ValueError(f"Unexpected input_ids dimension: {input_ids.dim()}")

        text_feats = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = self.text_proj(text_feats)
        text_features_flat = text_feats.view(-1, self.args.proj_dim)

        # 3. Compute Logits/Similarity
        compute_tokens = image_tokens
        if not self.criterion.use_vision_cls_token:
            compute_tokens = compute_tokens[:, 1:]

        # Get temperature
        if self.criterion.attn_temperature is not None:
            temp = self.criterion.attn_temperature.exp()
        else:
            temp = self.criterion.loss_temperature.exp()

        # Compute logits and attention
        logits, sim_scores = self.criterion.similarity_logit(
            text_features_flat, 
            compute_tokens, 
            temperature=temp
        )

        # # Softmax for attention weights
        # attn_weights = torch.softmax(sim_scores, dim=-1)

        # Remove CLS token from attention weights if it was used
        # if self.criterion.use_vision_cls_token:
        #     attn_weights = attn_weights[:, :, 1:]


        if self.criterion.use_vision_cls_token:
            sim_scores = sim_scores[:, :, 1:]

        # Scale logits
        logits = logits / temp

        return {
            "logits": logits,
            "similarity_scores": sim_scores
        }

    def compute_logits_for_text_features(
        self,
        pixel_values,
        encoded_key_phrases,
        **kwargs,
    ):
        # 1. Vision Forward
        image_tokens = self.vision_encoder(pixel_values)
        image_tokens = self.vision_proj(image_tokens)

        # 2. Text Forward
        # encoded_key_phrases is expected to be a list containing the batch encoding
        if isinstance(encoded_key_phrases, list):
            batch_encoding = encoded_key_phrases[0]
        else:
            batch_encoding = encoded_key_phrases

        device = pixel_values.device
        input_ids = batch_encoding["input_ids"].to(device)
        attention_mask = batch_encoding["attention_mask"].to(device)

        # TextEncoder expects [Batch, Num_Entities, Seq_Len]
        # Treat input [N, Seq_Len] as [1, N, Seq_Len]
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(0) 
            attention_mask = attention_mask.unsqueeze(0)

        text_feats = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = self.text_proj(text_feats)
        text_features_flat = text_feats.view(-1, self.args.proj_dim)

        # 3. Compute Logits/Similarity
        compute_tokens = image_tokens
        if not self.criterion.use_vision_cls_token:
            compute_tokens = compute_tokens[:, 1:]

        # Get temperature
        # V2 uses attn_temperature
        if hasattr(self.criterion, 'attn_temperature') and self.criterion.attn_temperature is not None:
            temp = self.criterion.attn_temperature.exp()
        else:
            temp = 1.0

        # Compute logits and attention
        logits, sim_scores = self.criterion.similarity_logit(
            text_features_flat, 
            compute_tokens, 
            temperature=temp
        )

        if self.criterion.use_vision_cls_token:
            sim_scores = sim_scores[:, :, 1:]

        # V2 Logic: Apply SigLIP scale and bias
        # logits = logits * self.logit_scale.exp() + self.logit_bias
        logits = logits * self.criterion.logit_scale.exp() + self.criterion.logit_bias

        return {
            "logits": logits,
            "similarity_scores": sim_scores
        }
