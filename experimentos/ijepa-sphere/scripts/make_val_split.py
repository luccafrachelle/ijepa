"""Stratified 10% internal validation split of train (seed 0)."""
import os, sys
import numpy as np

os.environ.setdefault("HF_HOME", os.path.abspath("data/hf"))


def make_val_ids(labels, frac=0.1, seed=0):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        k = int(round(frac * len(idx)))
        out.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(out))


def main(out="data/val_interno.txt"):
    from datasets import load_dataset
    ds = load_dataset("clane9/imagenet-100", cache_dir="data/imagenet100")["train"]
    ids = make_val_ids(ds["label"])
    with open(out, "w") as f:
        f.write("\n".join(str(i) for i in ids) + "\n")
    print("wrote", out, len(ids))


if __name__ == "__main__":
    main(*sys.argv[1:])
