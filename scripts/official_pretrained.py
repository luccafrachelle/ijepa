"""Run the OFFICIAL SPHERE-JEPA run_pretrained.main unmodified, except for checkpointing.

The official make_trainer uses ModelCheckpoint(save_last=True, save_top_k=0), which in Lightning 2.6
writes last.ckpt only at the end of training: a 21 h run cannot be resumed after an interruption.
This wrapper adds a per-epoch last.ckpt (same path, so the official --checkpoint resume works).
Training math, data, augmentations and evaluation are untouched. Run from third_party/SPHEREJEPA.
"""
import sys
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.insert(0, ".")
import run_utils  # noqa: E402
import run_pretrained  # noqa: E402



class PerEpochCheckpoint(ModelCheckpoint):
    """The official ModelCheckpoint(dirpath, save_last=True, save_top_k=0) plus a save every epoch."""
    def __init__(self, **kw):
        kw.update(save_top_k=1, every_n_epochs=1, filename="{epoch:03d}")
        super().__init__(**kw)


run_utils.ModelCheckpoint = PerEpochCheckpoint  # make_trainer looks it up in run_utils

if __name__ == "__main__":
    run_pretrained.main()
