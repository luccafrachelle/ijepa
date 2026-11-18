import sys, time, os
os.environ.setdefault("HF_HOME", os.path.abspath("data/hf"))
import torch
from torch.utils.data import DataLoader
from ijepa_sphere.data import load_imagenet100, TransformDataset, ijepa_train_transform

ds = TransformDataset(load_imagenet100()["train"], ijepa_train_transform())
for w in [int(x) for x in (sys.argv[1:] or ["8", "12", "16"])]:
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=w, pin_memory=True,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)
    it = iter(dl)
    for _ in range(5): next(it)  # warmup
    t = time.time(); n = 0
    for _ in range(50):
        x, y = next(it); n += len(x)
    print(f"workers={w}: {n/(time.time()-t):.0f} img/s", flush=True)
    del it, dl
