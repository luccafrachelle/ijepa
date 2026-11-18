"""Verify ImageNet-100 (HF), decode every image, write data/manifest.json."""
import hashlib, json, os, subprocess, sys
from collections import Counter
from multiprocessing import Pool

os.environ.setdefault("HF_HOME", os.path.abspath("data/hf"))
ROOT = "data/imagenet100"
SPLITS = {"train": "train", "validation": "validation"}


def _load():
    from datasets import load_dataset
    return load_dataset("clane9/imagenet-100", cache_dir=ROOT)


def _work(args):
    split, lo, hi = args
    ds = _load()[split]
    bad = []
    for i in range(lo, hi):
        try:
            im = ds[i]["image"]
            im.load()
            im.convert("RGB")
        except Exception as e:  # noqa
            bad.append([split, i, repr(e)[:200]])
    return bad


def du(p):
    return int(subprocess.check_output(["du", "-sb", p]).split()[0])


def main(procs=16):
    ds = _load()
    man = {"splits": {}, "counts": {}, "per_class_counts": {}}
    h = hashlib.sha256()
    corrupt = []
    for split in SPLITS:
        d = ds[split]
        labels = d["label"]
        man["counts"][split] = len(d)
        c = Counter(labels)
        man["per_class_counts"][split] = {str(k): c[k] for k in sorted(c)}
        assert len(c) == 100, (split, len(c))
        for i, y in enumerate(labels):
            h.update(f"{split},{i},{y}\n".encode())
        step = 2000
        jobs = [(split, lo, min(lo + step, len(d))) for lo in range(0, len(d), step)]
        with Pool(procs) as p:
            for bad in p.imap_unordered(_work, jobs):
                corrupt += bad
        print(split, len(d), "classes", len(c), "min/max per class", min(c.values()), max(c.values()), flush=True)
    names = ds["train"].features["label"].names
    man.update(
        num_classes=len(names), label_names=names,
        num_corrupt=len(corrupt), corrupt_ids=sorted([x[:2] for x in corrupt]),
        corrupt_errors=corrupt,
        sha256_split_index_label=h.hexdigest(),
        disk_size_bytes={"data/imagenet100": du("data/imagenet100"), "data/hf": du("data/hf")},
        dataset_fingerprint={s: ds[s]._fingerprint for s in SPLITS},
        dataset_revision_dirs=sorted(os.listdir(os.path.join(
            ROOT, "clane9___imagenet-100", "default", "0.0.0"))),
        dataset_id="clane9/imagenet-100",
    )
    man["splits"] = list(SPLITS)
    os.makedirs("data", exist_ok=True)
    json.dump(man, open("data/manifest.json", "w"), indent=1)
    print("corrupt:", len(corrupt), "disk:", man["disk_size_bytes"], "sha:", man["sha256_split_index_label"])


if __name__ == "__main__":
    main()
