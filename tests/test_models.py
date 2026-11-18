import torch

from ijepa_sphere.masks import make_mask_collator
from ijepa_sphere.models import init_model
from ijepa_sphere.projection import make_projection_head


def test_vit_tiny_shapes_and_size():
    torch.manual_seed(0)
    enc, pred = init_model()
    assert enc.embed_dim == 192 and len(enc.blocks) == 12 and enc.num_heads == 3
    assert enc.pos_embed.shape == (1, 196, 192) and not enc.pos_embed.requires_grad  # no CLS, fixed sincos
    assert len(pred.predictor_blocks) == 6 and pred.predictor_embed.out_features == 192
    n_enc = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"\nencoder params {n_enc/1e6:.3f}M, predictor params "
          f"{sum(p.numel() for p in pred.parameters() if p.requires_grad)/1e6:.3f}M")
    assert 5.4e6 < n_enc < 5.8e6

    coll = make_mask_collator(224)
    (x, _), m_enc, m_pred = coll([(torch.randn(3, 224, 224), 0) for _ in range(2)])
    with torch.no_grad():
        full = enc(x)
        ctx = enc(x, m_enc)
        p = pred(ctx, m_enc, m_pred)
    assert full.shape == (2, 196, 192)
    assert ctx.shape == (2, m_enc[0].shape[1], 192)
    assert p.shape == (2 * 4, m_pred[0].shape[1], 192)


def test_projection_head_matches_official_structure():
    g = make_projection_head()
    lin = [m for m in g.modules() if isinstance(m, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in lin] == [(192, 2048), (2048, 2048), (2048, 256)]
    assert sum(isinstance(m, torch.nn.BatchNorm1d) for m in g.modules()) == 2
    assert sum(isinstance(m, torch.nn.GELU) for m in g.modules()) == 2
    assert g(torch.randn(4, 192)).shape == (4, 256)
