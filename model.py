"""
USCNet: Transformer-Based Multimodal Fusion with Segmentation Guidance
Adapted for NSCLC-Radiomics histology classification.

Architecture modules:
  (a) Visual and Textual Transformation (VTT)
  (b) ViT-UNetSeg Module
  (c) MSAF Feature Fusion Module (CEA + SMA)
  (d) Classification Module
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import VisionTransformer
import timm


# ---------------------------------------------------------------------------
# (a) Visual and Textual Transformation Module (VTT)
# ---------------------------------------------------------------------------

class PatchEmbedding3D(nn.Module):
    """3D Patch Embedding for volumetric CT data."""

    def __init__(self, in_channels=1, embed_dim=768, patch_size=(8, 16, 16)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = self.norm(x)
        return x, (D, H, W)


class EHRProjection(nn.Module):
    """Project EHR tabular features to transformer embedding space."""

    def __init__(self, ehr_dim=7, embed_dim=768, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.projection = nn.Sequential(
            nn.Linear(ehr_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, embed_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, ehr_dim)
        B = x.shape[0]
        x = self.projection(x)  # (B, embed_dim * num_tokens)
        x = x.view(B, self.num_tokens, -1)  # (B, num_tokens, embed_dim)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Encoder Block (e)
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """Standard Transformer encoder block: LN -> MSA -> LN -> MLP."""

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_drop, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class VisualTransformerEncoder(nn.Module):
    """
    ViT encoder for 3D volumes. Uses pretrained 2D ViT weights for the
    transformer blocks and wraps them with 3D patch embedding.
    Extracts features at layers 3, 6, 9, 12 for the UNetSeg decoder.
    """

    def __init__(
        self,
        in_channels=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        patch_size=(8, 16, 16),
        drop_rate=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbedding3D(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.extract_layers = [2, 5, 8, 11]  # 0-indexed: layers 3, 6, 9, 12

        # Will be lazily initialized on first forward pass to match sequence length
        self.pos_embed = None
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def load_pretrained_weights(self):
        """Load weights from pretrained ViT-B/16 for transformer blocks."""
        pretrained = timm.create_model("vit_base_patch16_224", pretrained=True)
        pretrained_sd = pretrained.state_dict()

        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}."
            block_sd = {}
            for k, v in pretrained_sd.items():
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    mapped = self._map_key(new_key)
                    if mapped is not None:
                        block_sd[mapped] = v

            if block_sd:
                missing, unexpected = block.load_state_dict(block_sd, strict=False)

        if "norm.weight" in pretrained_sd:
            self.norm.load_state_dict({
                "weight": pretrained_sd["norm.weight"],
                "bias": pretrained_sd["norm.bias"],
            })

    def _map_key(self, key):
        """Map timm ViT keys to our TransformerEncoderBlock keys."""
        mapping = {
            "norm1.weight": "norm1.weight",
            "norm1.bias": "norm1.bias",
            "attn.qkv.weight": None,  # needs special handling
            "attn.qkv.bias": None,
            "attn.proj.weight": "attn.out_proj.weight",
            "attn.proj.bias": "attn.out_proj.bias",
            "norm2.weight": "norm2.weight",
            "norm2.bias": "norm2.bias",
            "mlp.fc1.weight": "mlp.0.weight",
            "mlp.fc1.bias": "mlp.0.bias",
            "mlp.fc2.weight": "mlp.3.weight",
            "mlp.fc2.bias": "mlp.3.bias",
        }
        return mapping.get(key, None)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        x, spatial_shape = self.patch_embed(x)
        B, N, C = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        if self.pos_embed is None or self.pos_embed.shape[1] != x.shape[1]:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, x.shape[1], C, device=x.device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        hidden_states = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.extract_layers:
                hidden_states.append(x)

        x = self.norm(x)
        return x, hidden_states, spatial_shape


# ---------------------------------------------------------------------------
# (b) ViT-UNetSeg Module (CNN Decoder with skip connections)
# ---------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    """3D convolution block with instance norm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ViTUNetSegDecoder(nn.Module):
    """
    CNN Decoder that takes multi-scale features from ViT encoder
    (Z3, Z6, Z9, Z12) and produces segmentation output.
    """

    def __init__(self, embed_dim=768, seg_channels=1):
        super().__init__()
        self.reshape_dims = [384, 384, 384, 384]

        self.proj_layers = nn.ModuleList([
            nn.Linear(embed_dim, self.reshape_dims[i]) for i in range(4)
        ])

        self.up4 = nn.ConvTranspose3d(384, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(256 + 384, 256)

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(128 + 384, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(64 + 384, 64)

        self.final_conv = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, seg_channels, 1),
        )

    def _reshape_feature(self, feat, proj, spatial_shape):
        """Reshape transformer output to spatial feature map."""
        B = feat.shape[0]
        feat = feat[:, 1:]  # remove CLS token
        feat = proj(feat)   # (B, N, C')
        D, H, W = spatial_shape
        feat = feat.transpose(1, 2).view(B, -1, D, H, W)
        return feat

    def forward(self, hidden_states, spatial_shape):
        z3 = self._reshape_feature(hidden_states[0], self.proj_layers[0], spatial_shape)
        z6 = self._reshape_feature(hidden_states[1], self.proj_layers[1], spatial_shape)
        z9 = self._reshape_feature(hidden_states[2], self.proj_layers[2], spatial_shape)
        z12 = self._reshape_feature(hidden_states[3], self.proj_layers[3], spatial_shape)

        x = self.up4(z12)
        x = self._match_and_concat(x, z9)
        x = self.dec4(x)

        x = self.up3(x)
        x = self._match_and_concat(x, z6)
        x = self.dec3(x)

        x = self.up2(x)
        x = self._match_and_concat(x, z3)
        x = self.dec2(x)

        seg_features = x  # for MSAF module
        seg_out = self.final_conv(x)

        return seg_out, seg_features

    def _match_and_concat(self, x, skip):
        """Match spatial dimensions and concatenate."""
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)


# ---------------------------------------------------------------------------
# (f) Cross-Attention Mechanism (used by both CEA and SMA)
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Cross-attention: Q from one modality, K/V from another.
    Q = Image/CT features, K = Text/EHR features, V = Layer hidden states.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value):
        B, N_q, C = query.shape
        N_kv = key.shape[1]

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# (c) MSAF Feature Fusion Module
# ---------------------------------------------------------------------------

class CTEHRAttentionModule(nn.Module):
    """
    CT-EHR Attention Module (CEA).
    Fuses CT visual features with EHR text features via cross-attention.
    Q = CT features (image), K/V = EHR features (text).
    """

    def __init__(self, dim=768, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm_ct = nn.LayerNorm(dim)
        self.norm_ehr = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, attn_drop=drop, proj_drop=drop)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, ct_feat, ehr_feat):
        # ct_feat: (B, N_ct, dim), ehr_feat: (B, N_ehr, dim)
        q = self.norm_ct(ct_feat)
        kv = self.norm_ehr(ehr_feat)
        fused = ct_feat + self.cross_attn(q, kv, kv)
        fused = fused + self.mlp(self.norm_out(fused))
        return fused


class SegmentationMultimodalAttention(nn.Module):
    """
    Segmentation-Multimodal Attention Module (SMA).
    Incorporates segmentation feature maps into the fused CT-EHR representation.
    Q = seg features, K/V = CT-EHR fused features.
    """

    def __init__(self, seg_channels=64, fused_dim=768, out_dim=768, num_heads=8, drop=0.0):
        super().__init__()
        self.seg_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(1),
            nn.Linear(seg_channels, fused_dim),
            nn.LayerNorm(fused_dim),
        )
        self.seg_token_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
        )
        self.cross_attn = CrossAttention(fused_dim, num_heads, attn_drop=drop, proj_drop=drop)
        self.norm1 = nn.LayerNorm(fused_dim)
        self.norm2 = nn.LayerNorm(fused_dim)
        self.out_proj = nn.Linear(fused_dim, out_dim)

    def forward(self, seg_features, fused_features):
        # seg_features: (B, C_seg, D, H, W)
        # fused_features: (B, N, fused_dim)
        B = seg_features.shape[0]

        seg_emb = self.seg_proj(seg_features)  # (B, fused_dim)
        seg_tokens = self.seg_token_proj(seg_emb).unsqueeze(1)  # (B, 1, fused_dim)

        # Expand seg tokens to create query sequence
        seg_query = seg_tokens.expand(-1, fused_features.shape[1], -1)

        q = self.norm1(seg_query)
        kv = self.norm2(fused_features)
        out = fused_features + self.cross_attn(q, kv, kv)

        out = self.out_proj(out)
        return out


class MSAFFeatureFusion(nn.Module):
    """
    Complete MSAF Feature Fusion Module.
    CEA (CT-EHR Attention) -> SMA (Segmentation-Multimodal Attention)
    """

    def __init__(self, embed_dim=768, seg_channels=64, num_heads=8, drop=0.0):
        super().__init__()
        self.cea = CTEHRAttentionModule(embed_dim, num_heads, drop=drop)
        self.sma = SegmentationMultimodalAttention(
            seg_channels=seg_channels, fused_dim=embed_dim, out_dim=embed_dim,
            num_heads=num_heads, drop=drop,
        )

    def forward(self, ct_feat, ehr_feat, seg_features):
        ct_ehr_fused = self.cea(ct_feat, ehr_feat)
        multimodal_fused = self.sma(seg_features, ct_ehr_fused)
        return multimodal_fused


# ---------------------------------------------------------------------------
# (d) Classification Module
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Classification head: Linear -> Classification.
    Takes the fused multimodal representation and predicts histology class.
    """

    def __init__(self, in_dim=768, hidden_dim=256, num_classes=4, drop=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# ---------------------------------------------------------------------------
# Full USCNet Model
# ---------------------------------------------------------------------------

class USCNet(nn.Module):
    """
    USCNet: Transformer-Based Multimodal Fusion with Segmentation Guidance.

    Adapted for NSCLC-Radiomics dataset.
    Input: CT volume (B,1,D,H,W), EHR features (B,7)
    Output: Segmentation mask (B,1,D,H,W), Classification logits (B,num_classes)
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: tuple = (8, 16, 16),
        ehr_dim: int = 7,
        ehr_tokens: int = 4,
        num_classes: int = 4,
        drop_rate: float = 0.1,
        seg_channels: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # (a) Visual Transformation
        self.vit_encoder = VisualTransformerEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            drop_rate=drop_rate,
        )

        # (a) Textual Transformation (EHR Projection)
        self.ehr_projection = EHRProjection(
            ehr_dim=ehr_dim, embed_dim=embed_dim, num_tokens=ehr_tokens,
        )

        # (b) ViT-UNetSeg Decoder
        self.seg_decoder = ViTUNetSegDecoder(
            embed_dim=embed_dim, seg_channels=1,
        )

        # (c) MSAF Feature Fusion
        self.msaf = MSAFFeatureFusion(
            embed_dim=embed_dim, seg_channels=seg_channels,
            num_heads=num_heads, drop=drop_rate,
        )

        # (d) Classification Head
        self.classifier = ClassificationHead(
            in_dim=embed_dim, hidden_dim=256, num_classes=num_classes, drop=drop_rate,
        )

    def load_pretrained(self):
        """Load pretrained weights for ViT encoder blocks."""
        self.vit_encoder.load_pretrained_weights()

    def freeze_backbone(self):
        """Freeze all pretrained backbone weights."""
        for param in self.vit_encoder.blocks.parameters():
            param.requires_grad = False
        for param in self.vit_encoder.norm.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone weights for fine-tuning."""
        for param in self.vit_encoder.blocks.parameters():
            param.requires_grad = True
        for param in self.vit_encoder.norm.parameters():
            param.requires_grad = True

    def forward(self, ct, ehr):
        """
        Args:
            ct: (B, 1, D, H, W) - CT volume
            ehr: (B, ehr_dim) - EHR features
        Returns:
            seg_out: (B, 1, D, H, W) - Segmentation prediction
            cls_out: (B, num_classes) - Classification logits
        """
        # (a) Visual transformation - ViT Encoder
        vit_out, hidden_states, spatial_shape = self.vit_encoder(ct)

        # (a) Textual transformation - EHR Projection
        ehr_tokens = self.ehr_projection(ehr)  # (B, num_tokens, embed_dim)

        # (b) ViT-UNetSeg: Decode segmentation from multi-scale features
        seg_out, seg_features = self.seg_decoder(hidden_states, spatial_shape)

        # Upsample segmentation to input resolution
        seg_out = F.interpolate(
            seg_out, size=ct.shape[2:], mode="trilinear", align_corners=False,
        )

        # Average pool the ViT output (excluding CLS) as CT representation
        ct_feat = vit_out[:, 1:]  # remove CLS, (B, N, embed_dim)

        # (c) MSAF: Fuse CT, EHR, and segmentation features
        fused = self.msaf(ct_feat, ehr_tokens, seg_features)

        # Global average pooling over sequence dimension
        fused_cls = fused.mean(dim=1)  # (B, embed_dim)

        # (d) Classification
        cls_out = self.classifier(fused_cls)

        return seg_out, cls_out


def build_model(args) -> USCNet:
    """Factory function to build the USCNet model."""
    model = USCNet(
        in_channels=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        patch_size=(8, 16, 16),
        ehr_dim=7,
        ehr_tokens=4,
        num_classes=4,
        drop_rate=getattr(args, "drop_rate", 0.1),
        seg_channels=64,
    )
    model.load_pretrained()
    return model
