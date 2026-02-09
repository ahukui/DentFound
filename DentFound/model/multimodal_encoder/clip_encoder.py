import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from typing import List, Dict, Tuple, Optional
import math
from typing import Optional, Sequence, Tuple, Union
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention supporting both self-attention and cross-attention.
    
    For self-attention: query, key, value all come from the same input.
    For cross-attention: query comes from one modality, key and value from another.
    
    Attention is computed as: softmax(Q K^T / sqrt(d)) V
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(d)

        # MLP_Q, MLP_K, MLP_V: linear projections for generating Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,       # (B, N_q, D)
        key: torch.Tensor,         # (B, N_kv, D)
        value: torch.Tensor,       # (B, N_kv, D)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N_q, D = query.shape
        N_kv = key.shape[1]

        # MLP_Q(query), MLP_K(key), MLP_V(value)
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention: softmax(Q K^T / sqrt(d)) V
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class InteractionBlock(nn.Module):
    """
    Interaction block that aligns mask and visual embeddings and injects
    tooth positional and semantic information into visual representations.

    Composed of:
        - Linear projection layer
        - Multi-Head Self-Attention (MHSA) for mask embeddings
        - Multi-Head Cross-Attention (MHCA) for fusing mask info into visual embeddings
        - Feed-Forward Network (FFN)

    Equations:
        E'_mask = E_mask + MHSA(MLP_K(E_mask), MLP_Q(E_mask), MLP_V(E_mask))
        E'_img  = E_img  + MHCA(MLP_K(E_img), FFN(E'_mask), MLP_V(E_img))

    In MHCA:
        - Keys (K) come from E_img
        - Queries (Q) come from FFN(E'_mask)
        - Values (V) come from E_img
    """

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # MHSA: self-attention for mask embeddings (Eq. 3)
        self.mhsa = MultiHeadAttention(
            dim, num_heads, attn_drop=dropout, proj_drop=dropout
        )

        # MHCA: cross-attention to inject mask info into visual embeddings (Eq. 4)
        self.mhca = MultiHeadAttention(
            dim, num_heads, attn_drop=dropout, proj_drop=dropout
        )

        # Layer norms
        self.norm_mask = nn.LayerNorm(dim)       # Pre-norm for mask self-attention
        self.norm_img_k = nn.LayerNorm(dim)      # Pre-norm for image keys in cross-attention
        self.norm_img_v = nn.LayerNorm(dim)      # Pre-norm for image values in cross-attention

        # FFN applied to E'_mask before serving as query in MHCA (Eq. 4)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(dim)  # Pre-norm for FFN

    def forward(
        self,
        img_features: torch.Tensor,   # E_img:  (B, N_img, D)
        mask_features: torch.Tensor,   # E_mask: (B, N_mask, D)
    ) -> torch.Tensor:
        """
        Args:
            img_features:  Visual embeddings E_img, shape (B, N_img, D)
            mask_features: Mask embeddings E_mask, shape (B, N_mask, D)

        Returns:
            E'_img: Position- and semantic-aware visual embeddings, shape (B, N_img, D)
        """

        # === Eq. 3: E'_mask = E_mask + MHSA(MLP_K(E_mask), MLP_Q(E_mask), MLP_V(E_mask)) ===
        mask_norm = self.norm_mask(mask_features)
        # Self-attention: Q, K, V all from mask embeddings
        mask_attn = self.mhsa(
            query=mask_norm,  # MLP_Q(E_mask)
            key=mask_norm,    # MLP_K(E_mask)
            value=mask_norm,  # MLP_V(E_mask)
        )
        e_mask_prime = mask_features + mask_attn  # Residual connection

        # === Eq. 4: E'_img = E_img + MHCA(MLP_K(E_img), FFN(E'_mask), MLP_V(E_img)) ===
        # FFN(E'_mask) serves as query
        query_for_cross = self.ffn(self.norm_ffn(e_mask_prime))

        # E_img provides keys and values
        img_norm = self.norm_img_k(img_features)

        # Cross-attention: Q from FFN(E'_mask), K and V from E_img
        cross_attn = self.mhca(
            query=query_for_cross,  # FFN(E'_mask)
            key=img_norm,           # MLP_K(E_img)
            value=img_norm,         # MLP_V(E_img)
        )
        e_img_prime = img_features + cross_attn  # Residual connection

        return e_img_prime


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        # InteractionBlock hyperparameters
        self.interaction_num_heads = getattr(args, 'interaction_num_heads', 8)
        self.interaction_mlp_ratio = getattr(args, 'interaction_mlp_ratio', 4)
        self.interaction_dropout = getattr(args, 'interaction_dropout', 0.1)
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        # Initialize the InteractionBlock with the encoder's hidden size
        self.interaction_block = InteractionBlock(
            dim=self.config.hidden_size,
            num_heads=self.interaction_num_heads,
            mlp_ratio=self.interaction_mlp_ratio,
            dropout=self.interaction_dropout,
        )

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        # image and mask are concated into one image
        image = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
        mask = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
        # Frozen encoder: no gradients for CLIP
        with torch.no_grad():
            image_forward_outs = self.vision_tower(image.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(image.dtype)

            mask_forward_outs = self.vision_tower(mask.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            mask_features = self.feature_select(mask_forward_outs).to(mask.dtype)

        # Trainable interaction block: gradients enabled
        fused_features = self.interaction_block(image_features, mask_features)
        return fused_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        # InteractionBlock hyperparameters
        self.interaction_num_heads = getattr(args, 'interaction_num_heads', 8)
        self.interaction_mlp_ratio = getattr(args, 'interaction_mlp_ratio', 4)
        self.interaction_dropout = getattr(args, 'interaction_dropout', 0.1)
        
        
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        # Initialize the InteractionBlock with the encoder's hidden size
        self.interaction_block = InteractionBlock(
            dim=self.config.hidden_size * len(self.s2_scales),
            num_heads=self.interaction_num_heads,
            mlp_ratio=self.interaction_mlp_ratio,
            dropout=self.interaction_dropout,
        )

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        image = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
        mask = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
        # Frozen encoder: no gradients for CLIP multiscale forward
        with torch.no_grad():
            image_feature = self.multiscale_forward(self.forward_feature, image, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
            mask_feature = self.multiscale_forward(self.forward_feature, mask, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
        # Trainable interaction block: gradients enabled
        fused_features = self.interaction_block(image_feature, mask_feature)
        return fused_features
    

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
