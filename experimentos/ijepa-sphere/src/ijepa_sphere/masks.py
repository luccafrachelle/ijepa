"""Multi-block masking (I-JEPA).

`MaskCollator` is the official class, unchanged (see `_multiblock.py`).
`make_mask_collator()` builds it with the values of the official config
third_party/ijepa/configs/in1k_vith14_ep300.yaml (`mask:` section), except
`patch_size`, which is 16 here (ViT-Tiny/16) instead of 14.

Collator output (used as a DataLoader `collate_fn`) for a batch of B samples:
    (collated_batch, masks_enc, masks_pred)
    collated_batch : default_collate(batch), e.g. (imgs[B,3,H,W], labels[B])
    masks_enc      : list (len nenc=1) of LongTensor[B, K_enc]  patch indices kept by the context encoder
    masks_pred     : list (len npred=4) of LongTensor[B, K_pred] patch indices of each target block
Indices are in [0, (H/16)*(W/16)) (196 for 224 px). Context indices exclude every
target block (allow_overlap=False). Lengths are truncated to the batch minimum.
"""
from ijepa_sphere._multiblock import MaskCollator
from ijepa_sphere.tensors import apply_masks, repeat_interleave_batch

# Official I-JEPA mask config (in1k_vith14_ep300.yaml), patch adapted to 16.
IJEPA_MASK_CFG = dict(
    patch_size=16,
    enc_mask_scale=(0.85, 1.0),
    pred_mask_scale=(0.15, 0.2),
    aspect_ratio=(0.75, 1.5),
    nenc=1,
    npred=4,
    min_keep=10,
    allow_overlap=False,
)


def make_mask_collator(input_size=224, **overrides):
    """Official MaskCollator with I-JEPA defaults; keyword overrides allowed."""
    cfg = {**IJEPA_MASK_CFG, **overrides}
    return MaskCollator(input_size=input_size, **cfg)


__all__ = ["MaskCollator", "make_mask_collator", "IJEPA_MASK_CFG",
           "apply_masks", "repeat_interleave_batch"]
