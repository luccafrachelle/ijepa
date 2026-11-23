"""Export HF ImageNet-100 to ImageFolder (original resolution) for source='folder'.
Output: data/imagenet100_folder/{train,val}/<class_idx:03d>_<wnid-less name>/<idx>.JPEG
Class dir names sort in HF label order, so ImageFolder labels == HF labels.
NOT run automatically."""
import io, os
from multiprocessing import Pool

os.environ.setdefault("HF_HOME", os.path.abspath("data/hf"))
OUT = "data/imagenet100_folder"


def _work(args):
    split, hf_split, lo, hi, names = args
    from datasets import load_dataset
    ds = load_dataset("clane9/imagenet-100", cache_dir="data/imagenet100")[hf_split]
    for i in range(lo, hi):
        s = ds[i]
        d = os.path.join(OUT, split, f"{s['label']:03d}")
        os.makedirs(d, exist_ok=True)
        s["image"].convert("RGB").save(os.path.join(d, f"{i}.JPEG"), quality=95)


def main(procs=16):
    from datasets import load_dataset
    ds = load_dataset("clane9/imagenet-100", cache_dir="data/imagenet100")
    jobs = []
    for split, hf in (("train", "train"), ("val", "validation")):
        n = len(ds[hf])
        jobs += [(split, hf, lo, min(lo + 2000, n), None) for lo in range(0, n, 2000)]
    with Pool(procs) as p:
        for _ in p.imap_unordered(_work, jobs):
            pass


if __name__ == "__main__":
    main()
