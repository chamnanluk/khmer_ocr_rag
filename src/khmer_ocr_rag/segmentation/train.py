from __future__ import annotations


def train_boundary_epoch(segmenter, loader, optimizer, device="cpu", pos_weight: float | None = None):
    torch = segmenter.torch
    model = segmenter.model.to(device)
    model.train()
    weight = None if pos_weight is None else torch.tensor([pos_weight], device=device)
    total = 0.0
    batches = 0
    for batch in loader:
        ids = batch["ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = segmenter.logits(ids)
        loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none", pos_weight=weight
        )
        loss = loss_raw[mask].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        batches += 1
    return total / max(1, batches)
