"""Encoder / predictor construction, following ijepa src/helper.py::init_model.

ViT-Tiny/16: embed 192, depth 12, 3 heads, no CLS token, fixed 2D sin-cos
pos-embed (official `vit_tiny`). Predictor: official `vit_predictor`
(narrow ViT), here depth 6 and width 192 (PLAN §5.1), heads = encoder heads (3).
"""
import torch

from ijepa_sphere import vision_transformer as vit
from ijepa_sphere.tensors import trunc_normal_


def init_model(model_name="vit_tiny", patch_size=16, crop_size=224,
               pred_depth=6, pred_emb_dim=192):
    """Same as ijepa helper.init_model (without device move / logging).

    Note: like the official code, the extra `init_weights` pass re-initialises
    every Linear with trunc_normal(std=0.02), which overrides the
    `fix_init_weight` depth rescaling done in the constructors. Kept on purpose.
    Returns (encoder, predictor).
    """
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    return encoder, predictor
