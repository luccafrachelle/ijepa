"""Evaluation protocol (PLAN §5.1/§7): frozen features, weighted k-NN, linear probe.

`extract_features`, `weighted_knn` and `linear_probe` are copied verbatim from
darktanuki/SPHEREJEPA @ 9a55566, evaluation.py (no license; academic use, see PLAN §2).
They are NOT imported from third_party at runtime.

Official protocol details (read from that file):
  * k-NN: cosine similarity, k=20, T=0.07, weights exp((s - s_max)/T).
  * linear probe: nn.Linear on raw (unnormalized) features, SGD momentum 0.9,
    no weight decay, constant lr = lr * batch_size / 256 (lr=0.1, batch_size = the
    config batch_size, 256 -> effective lr 0.1), 90 epochs, no lr schedule,
    no augmentation (features are extracted once with the eval transform; the
    official `evaluate_classification` also swaps the train transform for the
    deterministic val transform).

Project addition: `evaluate(model, train_ds, eval_ds, cfg)` with features =
average of the target encoder's output tokens (`model.features(pool='avg')`).
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


# ----------------------------- verbatim (SPHEREJEPA @ 9a55566) -----------------------------
@torch.inference_mode()
def extract_features(encoder, dataset, batch_size, num_workers, device, max_batches=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, persistent_workers=num_workers > 0)
    encoder = encoder.to(device).eval()
    features, labels = [], []
    for index, (views, target) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images = views[0] if isinstance(views, (list, tuple)) else views
        features.append(encoder(images.to(device)).float().cpu())
        labels.append(target.long().cpu())
    if not features:
        raise ValueError("No images were available for feature extraction.")
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def weighted_knn(train_x, train_y, eval_x, eval_y, num_classes, k=20,
                 temperature=0.07, query_batch_size=256):
    if k < 1 or temperature <= 0:
        raise ValueError("k and temperature must be positive.")
    train_x = F.normalize(train_x.float(), dim=1)
    eval_x = F.normalize(eval_x.float(), dim=1)
    k = min(k, len(train_x))
    correct = 0
    for start in range(0, len(eval_x), query_batch_size):
        query = eval_x[start:start + query_batch_size]
        similarity, indices = (query @ train_x.T).topk(k, dim=1)
        neighbors = train_y[indices]
        weights = ((similarity - similarity[:, :1]) / temperature).exp()
        votes = torch.zeros(len(query), num_classes)
        votes.scatter_add_(1, neighbors, weights)
        correct += int((votes.argmax(dim=1) == eval_y[start:start + len(query)]).sum())
    return correct / len(eval_y)


def linear_probe(train_x, train_y, eval_x, eval_y, num_classes, device,
                 epochs=90, batch_size=256, lr=0.1, seed=0):
    if epochs < 1:
        raise ValueError("Linear-probe epochs must be positive.")
    torch.manual_seed(seed)
    classifier = nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr * batch_size / 256,
                                momentum=0.9)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        classifier.train()
        for indices in torch.randperm(len(train_y), generator=generator).split(batch_size):
            logits = classifier(train_x[indices].to(device))
            loss = F.cross_entropy(logits, train_y[indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    classifier.eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(eval_y), batch_size):
            predictions = classifier(eval_x[start:start + batch_size].to(device)).argmax(1)
            correct += int((predictions.cpu() == eval_y[start:start + batch_size]).sum())
    return correct / len(eval_y)


# ------------------------------- end of verbatim copy -------------------------------------


class AvgPoolFeatures(nn.Module):
    """Wraps an IJEPA model so the official `extract_features` sees `encoder(images) -> [B, D]`:
    average of the target encoder's output tokens (I-JEPA eval protocol)."""

    def __init__(self, model, autocast_bf16=False):
        super().__init__()
        self.model = model
        self.autocast_bf16 = autocast_bf16

    def forward(self, images):
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16,
                            enabled=self.autocast_bf16 and images.device.type == "cuda"):
            return self.model.features(images, pool="avg").float()


def evaluate(model, train_ds, eval_ds, cfg, device=None, smoke=False, probe=True):
    """kNN (+ linear probe) with frozen avg-pooled target-encoder features.

    train_ds / eval_ds must yield (image_tensor, label) with the eval transform.
    cfg keys (all optional): batch_size (256), num_workers (12), knn_k (20),
    knn_temperature (0.07), linear_probe_epochs (90), linear_probe_lr (0.1),
    seed (0), num_classes (100), feature_bf16 (False).
    smoke=True mirrors the official smoke mode: 2 batches per split, 1 probe epoch.
    Returns dict: knn_top1, [linear_top1], n_train, n_eval. Leaves the model in eval mode.
    """
    device = device or next(model.parameters()).device
    bs = cfg.get("batch_size", 256)
    max_batches = 2 if smoke else None
    enc = AvgPoolFeatures(model, cfg.get("feature_bf16", False))
    options = (bs, cfg.get("num_workers", 12), device, max_batches)
    train_x, train_y = extract_features(enc, train_ds, *options)
    eval_x, eval_y = extract_features(enc, eval_ds, *options)
    nc = cfg.get("num_classes", 100)
    out = {"n_train": len(train_y), "n_eval": len(eval_y)}
    out["knn_top1"] = weighted_knn(train_x, train_y, eval_x, eval_y, nc,
                                   k=cfg.get("knn_k", 20),
                                   temperature=cfg.get("knn_temperature", 0.07))
    if probe:
        out["linear_top1"] = linear_probe(
            train_x, train_y, eval_x, eval_y, nc, device,
            epochs=1 if smoke else cfg.get("linear_probe_epochs", 90),
            batch_size=bs, lr=cfg.get("linear_probe_lr", 0.1), seed=cfg.get("seed", 0))
    return out
