import pytest
import torch

from ijepa_sphere.jepa import IJEPA
from ijepa_sphere.masks import make_mask_collator


def _batch(B=4, seed=0):
    torch.manual_seed(seed)
    coll = make_mask_collator(224)
    (x, _), m_enc, m_pred = coll([(torch.randn(3, 224, 224), 0) for _ in range(B)])
    return x, m_enc, m_pred


def _trainable_ids(model):
    return {id(p) for p in model.parameters() if p.requires_grad}


@pytest.mark.parametrize("reg", ["none", "sigreg", "susreg"])
def test_ema_off_shares_weights_and_target_has_no_grad(reg):
    model = IJEPA(ema=False, reg=reg)
    assert model.target_encoder is None and model.target is model.encoder
    x, m_enc, m_pred = _batch()
    h = model.forward_target(x, m_enc, m_pred)
    assert not h.requires_grad and h.grad_fn is None
    out = model(x, m_enc, m_pred)
    assert out["loss"].requires_grad
    out["loss"].backward()
    assert model.encoder.patch_embed.proj.weight.grad is not None
    if reg == "none":
        assert model.projector is None and float(out["loss_reg"]) == 0.0
    else:
        assert model.projector[0].weight.grad is not None
        assert torch.allclose(out["loss"], out["loss_pred"] + 0.05 * out["loss_reg"])


def test_ema_off_gradient_equals_detached_target():
    """Gradient with shared weights == gradient with a frozen copy as target (m=0)."""
    import copy
    torch.manual_seed(0)
    model = IJEPA(ema=False, reg="none")
    x, m_enc, m_pred = _batch()
    model(x, m_enc, m_pred)["loss"].backward()
    g_shared = model.encoder.blocks[0].attn.qkv.weight.grad.clone()

    model.zero_grad()
    frozen = copy.deepcopy(model.encoder).requires_grad_(False)
    model.use_ema, model.target_encoder = True, frozen
    model(x, m_enc, m_pred)["loss"].backward()
    assert torch.allclose(g_shared, model.encoder.blocks[0].attn.qkv.weight.grad, atol=1e-6)


def test_ema_on_target_frozen_and_update():
    model = IJEPA(ema=True, reg="susreg")
    x, m_enc, m_pred = _batch()
    model(x, m_enc, m_pred)["loss"].backward()
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert not any(p.requires_grad for p in model.target_encoder.parameters())
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    q = model.encoder.blocks[0].mlp.fc1.weight
    k = model.target_encoder.blocks[0].mlp.fc1.weight
    k0 = k.clone()
    model.ema_update(0.996)
    assert torch.allclose(k, 0.996 * k0 + 0.004 * q)


def test_param_groups_cover_trainables_once():
    for reg in ["none", "susreg"]:
        model = IJEPA(ema=True, reg=reg)
        groups = model.param_groups()
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids)) and set(ids) == _trainable_ids(model)
        assert groups[1]["WD_exclude"] and groups[1]["weight_decay"] == 0
        assert all(p.ndim > 1 for p in groups[0]["params"])


def test_features_shape():
    model = IJEPA(ema=True)
    x, _, _ = _batch(B=2)
    assert model.features(x).shape == (2, 192)
    assert model.features(x, pool="none").shape == (2, 196, 192)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_bf16_step():
    model = IJEPA(ema=True, reg="susreg").cuda()
    opt = torch.optim.AdamW(model.param_groups(), lr=1e-4)
    x, m_enc, m_pred = _batch(B=8)
    x = x.cuda(); m_enc = [m.cuda() for m in m_enc]; m_pred = [m.cuda() for m in m_pred]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, m_enc, m_pred)
    assert out["loss_reg"].dtype == torch.float32 and torch.isfinite(out["loss"])
    out["loss"].backward(); opt.step(); model.ema_update(0.996)
