from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainEpochResult:
    loss: float
    batches: int


def train_epoch(recognizer, loader, optimizer, device="cpu", boundary_flag: int = 1, grad_clip: float = 5.0):
    torch = recognizer.torch
    model = recognizer.model.to(device)
    model.train()
    total = 0.0
    batches = 0
    for batch in loader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        pixel_widths = batch["pixel_widths"].to(device)
        flags = torch.full((images.shape[0],), boundary_flag, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, _routing = model(images, flags)
        input_lengths = recognizer.output_lengths_from_pixel_widths(pixel_widths).clamp(max=logits.shape[1])
        loss = recognizer.ctc_loss(logits, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach().cpu())
        batches += 1
    return TrainEpochResult(total / max(1, batches), batches)


def greedy_decode(recognizer, vocab, images, pixel_widths, device="cpu", boundary_flag: int = 1):
    torch = recognizer.torch
    model = recognizer.model.to(device)
    model.eval()
    flags = torch.full((images.shape[0],), boundary_flag, dtype=torch.long, device=device)
    with torch.no_grad():
        logits, routing = model(images.to(device), flags)
        pred = logits.argmax(dim=-1).cpu()
    lengths = recognizer.output_lengths_from_pixel_widths(pixel_widths).clamp(max=pred.shape[1]).tolist()
    texts = [vocab.decode_ctc(row[:length].tolist()) for row, length in zip(pred, lengths)]
    return texts, routing.detach().cpu()
