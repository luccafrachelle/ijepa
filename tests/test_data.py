import os
import numpy as np
import pytest
import torch
from PIL import Image

from ijepa_sphere.data import (collapse_subset_ids, eval_transform,
                               ijepa_train_transform, split_train)

ROOT = os.path.join(os.path.dirname(__file__), "..")
needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "data/imagenet100")), reason="dataset absent")


def test_transform_shapes():
    img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    assert eval_transform()(img).shape == (3, 224, 224)
    assert ijepa_train_transform()(img).shape == (3, 224, 224)
    assert ijepa_train_transform(hflip=True)(img).dtype == torch.float32


def test_collapse_ids():
    a = collapse_subset_ids(1000, 100, 0)
    assert (a == collapse_subset_ids(1000, 100, 0)).all() and len(set(a)) == 100


@needs_data
def test_val_split_determinism_and_stratification(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("HF_HOME", os.path.abspath("data/hf"))
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import make_val_split
    from datasets import load_dataset
    labels = np.asarray(load_dataset("clane9/imagenet-100", cache_dir="data/imagenet100")["train"]["label"])
    a = make_val_split.make_val_ids(labels)
    assert (a == make_val_split.make_val_ids(labels)).all()
    f = tmp_path / "v.txt"
    make_val_split.main(str(f))
    assert f.read_text() == open("data/val_interno.txt").read()
    assert len(set(a)) == len(a) and (np.diff(a) > 0).all()
    for c in range(100):
        n = (labels == c).sum()
        assert (labels[a] == c).sum() == int(round(0.1 * n))
    from torch.utils.data import Subset
    tr, va = split_train(list(range(len(labels))), "data/val_interno.txt")
    assert set(tr.indices).isdisjoint(va.indices) and len(tr) + len(va) == len(labels)
