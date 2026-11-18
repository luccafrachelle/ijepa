# WarmupCosineSchedule and CosineWDSchedule: copied verbatim from
# facebookresearch/ijepa @ 52c1ae9, src/utils/schedulers.py
# (Copyright (c) Meta Platforms, Inc. and affiliates; CC BY-NC 4.0, see
# third_party/ijepa/LICENSE). No change to the logic.
#
# Added below the official classes: `momentum_schedule` / `momentum_at`, the
# official EMA momentum generator of ijepa src/train.py:
#     momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
#                           for i in range(int(ipe*num_epochs*ipe_scale)+1))
# and `build_schedules`, which wires them with this project's values
# (PLAN §5.1 + controller rulings):
#     start_lr = 0.2*lr, final_lr = 1e-6 (official ratios 2e-4/1e-3),
#     warmup = 10% of the epochs (PLAN), wd 0.04 -> 0.4 cosine (official),
#     EMA 0.996 -> 1.0 linear over ipe*epochs*ipe_scale (official, ipe_scale 1.0).

import math


class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd


# ---------------------------------------------------------------------------
# Project additions (not in the official schedulers.py)
# ---------------------------------------------------------------------------

def momentum_schedule(ema, ipe, num_epochs, ipe_scale=1.0):
    """Official ijepa train.py EMA momentum generator (linear ema[0] -> ema[1])."""
    return (ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
            for i in range(int(ipe * num_epochs * ipe_scale) + 1))


def momentum_at(i, ema, ipe, num_epochs, ipe_scale=1.0):
    """Value number i (0-based) of `momentum_schedule` (closed form, for resume)."""
    return ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)


def build_schedules(optimizer, ipe, opt_cfg):
    """Official init_opt wiring with this project's values.

    opt_cfg keys: epochs, lr, start_lr_ratio (0.2), final_lr (1e-6), warmup_frac (0.1),
    weight_decay (0.04), final_weight_decay (0.4), ipe_scale (1.0).
    Returns (lr_scheduler, wd_scheduler, total_steps). As in the official loop,
    call both .step() at the start of every iteration (before forward).
    """
    epochs = opt_cfg["epochs"]
    ipe_scale = opt_cfg.get("ipe_scale", 1.0)
    lr = float(opt_cfg["lr"])
    total = int(ipe_scale * epochs * ipe)
    warmup_steps = int(opt_cfg.get("warmup_frac", 0.1) * epochs * ipe)
    sched = WarmupCosineSchedule(
        optimizer, warmup_steps=warmup_steps,
        start_lr=float(opt_cfg.get("start_lr_ratio", 0.2)) * lr, ref_lr=lr,
        final_lr=float(opt_cfg.get("final_lr", 1e-6)), T_max=total)
    wd_sched = CosineWDSchedule(
        optimizer, ref_wd=float(opt_cfg.get("weight_decay", 0.04)),
        final_wd=float(opt_cfg.get("final_weight_decay", 0.4)), T_max=total)
    return sched, wd_sched, total
