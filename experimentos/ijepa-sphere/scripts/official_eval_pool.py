"""Re-evaluate an official SPHERE-JEPA/LeJEPA checkpoint with a chosen pooling (PLAN §6).

Wraps run_pretrained.main(--eval-only) unchanged, except that right before the official
evaluate_classification it sets the timm ViT's global_pool:
  token = CLS token (what the paper / training-end eval uses)
  avg   = mean of the patch tokens after the final norm, CLS excluded (as I-JEPA / our grid)

Run from third_party/SPHEREJEPA (see scripts/run_calib_pool.sh). Pass --output to a
directory other than runs/calib so the original evaluation.yaml is not overwritten.

    POOL=avg python scripts/official_eval_pool.py --config configs/X.yaml --eval-only \
        --checkpoint .../last.ckpt --root data/imagenet100 --output runs/calib_pool/avg
"""
import os
import sys

sys.path.insert(0, ".")
import run_pretrained  # noqa: E402

POOL = os.environ.get("POOL", "avg")
assert POOL in ("token", "avg"), POOL
_orig_eval = run_pretrained.evaluate_classification


def _eval_with_pool(model, data, config, smoke=False):
    model.f.global_pool = POOL
    print(f"[official_eval_pool] global_pool={model.f.global_pool} "
          f"num_prefix_tokens={model.f.num_prefix_tokens}", flush=True)
    return _orig_eval(model, data, config, smoke=smoke)


run_pretrained.evaluate_classification = _eval_with_pool

if __name__ == "__main__":
    run_pretrained.main()
