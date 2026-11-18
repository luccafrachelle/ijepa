import torch

from ijepa_sphere.masks import IJEPA_MASK_CFG, make_mask_collator

GRID = 14
N = GRID * GRID


def _is_rectangle(idx):
    r, c = idx // GRID, idx % GRID
    h = int(r.max() - r.min() + 1)
    w = int(c.max() - c.min() + 1)
    return h * w == len(idx), h, w


def test_mask_cfg_values():
    assert IJEPA_MASK_CFG == dict(patch_size=16, enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
                                  aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10, allow_overlap=False)


def test_mask_proportions_1000_samples():
    torch.manual_seed(0)
    coll = make_mask_collator(224)
    ctx_frac, tgt_frac, union_frac = [], [], []
    for _ in range(1000):
        (imgs,), m_enc, m_pred = coll([(torch.zeros(3, 2, 2),)])  # batch of 1 -> no truncation
        assert imgs.shape == (1, 3, 2, 2)
        assert len(m_enc) == 1 and len(m_pred) == 4
        ctx = m_enc[0][0]
        assert ctx.min() >= 0 and ctx.max() < N and len(set(ctx.tolist())) == len(ctx)
        tgt_union = set()
        sizes = set()
        for mp in m_pred:
            t = mp[0]
            ok, h, w = _is_rectangle(t)
            assert ok
            sizes.add((h, w))
            # one block size per batch (official: size sampled once per collate call)
            tgt_union |= set(t.tolist())
            # target blocks scale (0.15, 0.2) of 196 patches, up to rounding of h, w
            assert 0.12 * N <= len(t) <= 0.23 * N, len(t)
            tgt_frac.append(len(t) / N)
        assert len(sizes) == 1
        # context is disjoint from all targets (allow_overlap=False)
        assert not (set(ctx.tolist()) & tgt_union)
        ctx_frac.append(len(ctx) / N)
        union_frac.append(len(tgt_union) / N)
    ctx_mean = sum(ctx_frac) / len(ctx_frac)
    tgt_mean = sum(tgt_frac) / len(tgt_frac)
    uni_mean = sum(union_frac) / len(union_frac)
    print(f"\nmask stats (1000 samples): target block frac mean={tgt_mean:.3f}, "
          f"target union frac mean={uni_mean:.3f}, context frac mean={ctx_mean:.3f} "
          f"(min {min(ctx_frac):.3f}, max {max(ctx_frac):.3f})")
    assert 0.15 <= tgt_mean <= 0.20
    assert 0.2 <= ctx_mean <= 0.6


def test_batch_collation_truncates_to_min():
    torch.manual_seed(0)
    coll = make_mask_collator(224)
    batch = [(torch.zeros(3, 224, 224), i) for i in range(8)]
    (imgs, labels), m_enc, m_pred = coll(batch)
    assert imgs.shape == (8, 3, 224, 224) and labels.tolist() == list(range(8))
    assert m_enc[0].shape[0] == 8 and m_enc[0].dtype == torch.int64
    assert all(m.shape[0] == 8 and m.shape == m_pred[0].shape for m in m_pred)
