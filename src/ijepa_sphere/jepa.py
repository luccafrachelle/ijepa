"""I-JEPA model with optional EMA target and optional distribution regularizer.

Grid of PLAN §5.2:  ema in {True, False}  x  reg in {'none', 'sigreg', 'susreg'}.

Loss (PLAN §5.2):  L = L_pred + lambda_reg * L_reg
  L_pred: official I-JEPA (ijepa src/train.py): targets = layer_norm(target_encoder(imgs)),
          gathered at masks_pred, repeated per context mask; smooth_l1(pred, targets).
  L_reg : regularizer on projector(mean over visible context tokens of the context
          encoder)  ("R-pool"); 0 when reg='none'.

ema=True : target_encoder is a deepcopy of encoder, requires_grad=False, updated with
           `ema_update(m)` (official rule: p_k = m p_k + (1-m) p_q).
ema=False: the target IS the context encoder (shared weights), evaluated on the full
           image under torch.no_grad() (equivalent to m = 0). `target_encoder` is None.

Usage (training script):
    model = IJEPA(ema=True, reg='susreg', lambda_reg=0.05).cuda()
    groups = model.param_groups()                      # official wd / no-wd split
    opt = torch.optim.AdamW(groups)
    (imgs, _), masks_enc, masks_pred = batch           # from make_mask_collator()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(imgs, masks_enc, masks_pred)       # dict: loss, loss_pred, loss_reg
    out['loss'].backward(); opt.step(); opt.zero_grad()
    model.ema_update(m)                                # no-op when ema=False
    feats = model.features(imgs)                       # [B,192] eval features
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn

from ijepa_sphere.models import init_model
from ijepa_sphere.projection import make_projection_head
from ijepa_sphere.regularizers import build_regularizer
from ijepa_sphere.tensors import apply_masks, repeat_interleave_batch


class IJEPA(nn.Module):
    def __init__(self, ema=True, reg="none", lambda_reg=0.05,
                 model_name="vit_tiny", patch_size=16, crop_size=224,
                 pred_depth=6, pred_emb_dim=192,
                 proj_hidden_dim=2048, proj_dim=256, reg_kwargs=None):
        super().__init__()
        self.use_ema = bool(ema)
        self.reg_kind = "none" if reg is None else str(reg).lower()
        self.lambda_reg = float(lambda_reg)

        self.encoder, self.predictor = init_model(
            model_name=model_name, patch_size=patch_size, crop_size=crop_size,
            pred_depth=pred_depth, pred_emb_dim=pred_emb_dim)
        if self.use_ema:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        else:
            self.target_encoder = None

        self.regularizer = build_regularizer(self.reg_kind, dim=proj_dim, **(reg_kwargs or {}))
        if self.regularizer is not None:
            self.projector = make_projection_head(self.encoder.embed_dim, proj_hidden_dim, proj_dim)
        else:
            self.projector = None

    # ------------------------------------------------------------------ #
    @property
    def target(self):
        """Module producing targets / eval features (EMA copy or the encoder itself)."""
        return self.target_encoder if self.use_ema else self.encoder

    @torch.no_grad()
    def forward_target(self, imgs, masks_enc, masks_pred):
        """Official forward_target: [B*npred*nenc, K_pred, D] layer-normed targets."""
        h = self.target(imgs)
        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
        B = len(h)
        h = apply_masks(h, masks_pred)
        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
        return h

    def forward(self, imgs, masks_enc, masks_pred):
        """Returns dict with 'loss' (graph), 'loss_pred', 'loss_reg' (graph or zero tensor),
        and 'ctx_emb' [nenc*B, D] (pooled context-encoder tokens, detached) for logging."""
        if not isinstance(masks_enc, list):
            masks_enc = [masks_enc]
        if not isinstance(masks_pred, list):
            masks_pred = [masks_pred]

        h = self.forward_target(imgs, masks_enc, masks_pred)
        z_ctx = self.encoder(imgs, masks_enc)            # [nenc*B, K_enc, D]
        z = self.predictor(z_ctx, masks_enc, masks_pred)  # [npred*nenc*B, K_pred, D]
        loss_pred = F.smooth_l1_loss(z, h)

        pooled = z_ctx.mean(dim=1)                        # [nenc*B, D]
        if self.regularizer is not None:
            B = imgs.shape[0]
            proj = self.projector(pooled)                 # [nenc*B, proj_dim]
            loss_reg = self.regularizer(proj.view(len(masks_enc), B, -1))
        else:
            loss_reg = torch.zeros((), device=imgs.device)
        loss = loss_pred + self.lambda_reg * loss_reg
        return {"loss": loss, "loss_pred": loss_pred, "loss_reg": loss_reg,
                "ctx_emb": pooled.detach()}

    @torch.no_grad()
    def ema_update(self, m):
        """Official EMA step; no-op when ema=False."""
        if not self.use_ema:
            return
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

    @torch.no_grad()
    def features(self, imgs, pool="avg"):
        """Eval features: target encoder on full image. pool='avg' -> [B,D]; 'none' -> [B,N,D]."""
        h = self.target(imgs)
        return h.mean(dim=1) if pool == "avg" else h

    def param_groups(self):
        """Official ijepa init_opt grouping: weights (wd scheduled) vs bias/1-D
        params ('WD_exclude': True, weight_decay 0). Trainable modules: encoder,
        predictor and (if present) projector. Set 'lr'/'weight_decay' per step
        with the official WarmupCosineSchedule / CosineWDSchedule."""
        mods = [self.encoder, self.predictor] + ([self.projector] if self.projector is not None else [])
        decay, no_decay = [], []
        for mod in mods:
            for n, p in mod.named_parameters():
                if not p.requires_grad:
                    continue
                if ('bias' in n) or (len(p.shape) == 1):
                    no_decay.append(p)
                else:
                    decay.append(p)
        return [{"params": decay},
                {"params": no_decay, "WD_exclude": True, "weight_decay": 0}]
