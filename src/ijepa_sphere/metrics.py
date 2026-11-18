"""Collapse indicators (PLAN §7).

Which embedding is measured (same for all six cells, with or without projector):
the TARGET encoder run on the full image without augmentation (eval transform),
i.e. `model.features(imgs)`:
  * "pooled"  = average of the output tokens [N, 192] (= the eval features);
  * "tokens"  = individual output patch tokens [N*196, 192], subsampled.
With ema=False the target encoder IS the context encoder, so for C2/C4/C6 this is
also the trained encoder; with ema=True it is the EMA copy (what is evaluated).
The projector (only present when reg != 'none') is never measured.

Definitions:
  * mean_std_normalized: mean over dims of the per-dim std (over samples) of the
    L2-normalized embeddings. Collapse -> 0; isotropic on S^{d-1} -> ~1/sqrt(d).
  * effective_rank: exp(entropy(p)), p = s / sum(s), s the singular values of the
    (uncentered) embedding matrix (Roy & Vetterli 2007; RankMe, Garrido et al. 2023).
    Identical rows -> 1; isotropic -> close to d.
  * uniformity: log E_{i!=j} exp(-t ||z_i - z_j||^2), t=2, normalized z (Wang & Isola 2020).
  * alignment: E ||z(x1) - z(x2)||^2 (alpha=2), normalized z, x1/x2 two random
    I-JEPA crops (ijepa_train_transform) of the same image.
"""
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def mean_std_normalized(z):
    z = F.normalize(z.float(), dim=-1)
    return float(z.std(dim=0).mean())


def effective_rank(z, eps=1e-12):
    s = torch.linalg.svdvals(z.float())
    p = s / (s.sum() + eps)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def uniformity(z, t=2.0, max_n=4096, seed=0):
    z = F.normalize(z.float(), dim=-1)
    if len(z) > max_n:
        g = torch.Generator().manual_seed(seed)
        z = z[torch.randperm(len(z), generator=g)[:max_n]]
    sq = torch.pdist(z, p=2).pow(2)
    return float(torch.logsumexp(-t * sq, dim=0) - math.log(sq.numel()))


def alignment(z1, z2, alpha=2):
    z1, z2 = F.normalize(z1.float(), dim=-1), F.normalize(z2.float(), dim=-1)
    return float((z1 - z2).norm(dim=1).pow(alpha).mean())


class TwoCropDataset(Dataset):
    """(crop1, crop2) of the same image; the crops depend only on (seed, index),
    so they are identical at every evaluation and independent of worker seeding."""

    def __init__(self, ds, transform, seed=0):
        self.ds, self.transform, self.seed = ds, transform, seed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, _ = self.ds[i]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed * 1_000_003 + int(i))
            return self.transform(img), self.transform(img)


@torch.no_grad()
def _embed(model, loader, device, max_tokens_per_img, seed, bf16):
    pooled, tokens = [], []
    g = torch.Generator().manual_seed(seed)
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bf16 and device.type == "cuda"):
            h = model.features(x, pool="none").float()  # [B, N, D]
        pooled.append(h.mean(1).cpu())
        B, N, D = h.shape
        k = min(max_tokens_per_img, N)
        idx = torch.stack([torch.randperm(N, generator=g)[:k] for _ in range(B)]).to(h.device)
        tokens.append(torch.gather(h, 1, idx[..., None].expand(B, k, D)).reshape(-1, D).cpu())
    return torch.cat(pooled), torch.cat(tokens)


@torch.no_grad()
def collapse_report(model, subset_loader, pair_loader=None, device=None,
                    max_tokens_per_img=4, max_tokens=50000, seed=0, bf16=False):
    """Collapse indicators on a fixed subset (see module docstring).

    subset_loader: yields (imgs, labels) with the eval transform (no augmentation).
    pair_loader:   optional, yields (crop1, crop2) from `TwoCropDataset` for alignment.
    Returns dict: std_norm, erank_pooled, erank_tokens, uniformity, [alignment], n.
    The model is put in eval mode; the caller restores train mode.
    """
    device = device or next(model.parameters()).device
    model.eval()
    pooled, tokens = _embed(model, subset_loader, device, max_tokens_per_img, seed, bf16)
    if len(tokens) > max_tokens:
        g = torch.Generator().manual_seed(seed + 1)
        tokens = tokens[torch.randperm(len(tokens), generator=g)[:max_tokens]]
    out = {"n": len(pooled),
           "std_norm": mean_std_normalized(pooled),
           "erank_pooled": effective_rank(pooled),
           "erank_tokens": effective_rank(tokens),
           "uniformity": uniformity(pooled, seed=seed)}
    if pair_loader is not None:
        a, b = [], []
        for x1, x2 in pair_loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=bf16 and device.type == "cuda"):
                a.append(model.features(x1.to(device, non_blocking=True)).float().cpu())
                b.append(model.features(x2.to(device, non_blocking=True)).float().cpu())
        out["alignment"] = alignment(torch.cat(a), torch.cat(b))
    return out
