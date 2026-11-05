import itertools
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import Windows, load_manifest
from .models import ConditionalFlow, random_mask, fm_loss, reconstruct


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_name(value):
    return "cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # only own/trusted checkpoints
    model = ConditionalFlow(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def assert_compatible(checkpoint, manifest):
    if checkpoint["manifest"] != manifest:
        raise ValueError("Checkpoint and data manifest differ; do not mix caches or split statistics")


def corrupt_history(model, history, fraction, steps):
    """One-window self-generated history exposure; NOT full streaming rollout training."""
    if fraction <= 0:
        return history
    h = history.clone()
    use = torch.rand(len(h), device=h.device) < fraction
    if use.any():
        previous = h[use, -5:]
        m = random_mask(len(previous), 5, 1, h.device)
        earlier = h[use, :-5]
        h[use, -5:] = reconstruct(model, previous*m[..., None], m, earlier, steps)
    return h


def teacher(args):
    seed_all(args.seed)
    device = device_name(args.device)
    receiver, ckpt = load_model(args.receiver, device)
    manifest = load_manifest(args.data)
    assert_compatible(ckpt, manifest)
    dataset = Windows(args.data, "train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    combinations = list(itertools.combinations(range(5), args.drops))
    saved = {"history": [], "original": [], "target": []}
    # Distortion measured in unnormalized coefficient space (orthonormal DCT).
    std = torch.tensor(manifest["std"], device=device)
    total = 0
    for hist, original in loader:
        hist, original = hist.to(device), original.to(device)
        hist = corrupt_history(receiver, hist, args.rollout_fraction, args.receiver_steps)
        b = len(hist)
        common_noise = torch.randn_like(original)
        costs, masks = [], []
        for omitted in combinations:
            mask = torch.ones(b, 5, device=device)
            mask[:, list(omitted)] = 0
            rec = reconstruct(receiver, original*mask[..., None], mask, hist, args.receiver_steps, common_noise)
            costs.append(((rec-original)*std).square().mean((1, 2)))
            masks.append(mask)
        best = torch.stack(costs, 1).argmin(1)
        target_mask = torch.stack(masks, 1)[torch.arange(b, device=device), best]
        saved["history"].append(hist.cpu())
        saved["original"].append(original.cpu())
        # A dequantized continuous target for the mask flow; lower means omit.
        saved["target"].append((2*target_mask-1).cpu())
        total += b
        if total >= args.teacher_windows:
            break
    out = {k: torch.cat(v)[:args.teacher_windows] for k, v in saved.items()}
    out.update(manifest=manifest, drops=args.drops, receiver=str(args.receiver), seed=args.seed)
    path = Path(args.teacher)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"Saved {len(out['target'])} training-only teacher windows to {path}", flush=True)


def train(args):
    seed_all(args.seed)
    device = device_name(args.device)
    manifest = load_manifest(args.data)
    config = dict(dim=manifest["dim"], window=5, width=args.width, context=64, sender=args.stage == "sender")
    model = ConditionalFlow(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cuda_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=cuda_amp)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start, best = 0, float("inf")
    if args.resume:
        old = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert_compatible(old, manifest)
        if old["model_config"] != config:
            raise ValueError("Resume model configuration mismatch")
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        scaler.load_state_dict(old["scaler"])
        start, best = old["step"], old["best"]
        # DataLoader restarts from a seeded shuffle; resume is not bitwise replay.
    if args.stage == "sender":
        cache = torch.load(args.teacher, map_location="cpu", weights_only=False)
        assert_compatible(cache, manifest)
        if cache["drops"] != args.drops:
            raise ValueError("Teacher omission budget mismatch")
        dataset = torch.utils.data.TensorDataset(cache["history"], cache["original"], cache["target"])
        valid = None
    else:
        dataset = Windows(args.data, "train")
        valid = DataLoader(Windows(args.data, "valid"), batch_size=args.batch_size, num_workers=0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    iterator = iter(loader)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(json.dumps({"stage": args.stage, "device": device, "parameters": sum(p.numel() for p in model.parameters()), "windows": len(dataset)}), flush=True)
    for step in range(start+1, args.steps+1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model.train()
        if args.stage == "sender":
            hist, original, target = (x.to(device) for x in batch)
            target = target + .05 * torch.randn_like(target)
            mask = torch.full((len(hist), 5), 1-args.drops/5, device=device)
            observed = original
        else:
            hist, target = (x.to(device) for x in batch)
            hist = corrupt_history(model, hist, args.rollout_fraction, args.receiver_steps)
            # Mixture of intentional omissions and additional random channel loss.
            mask = random_mask(len(hist), 5, args.drops, device)
            mask *= (torch.rand_like(mask) > args.train_loss).float()
            observed = target * mask[..., None]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if cuda_amp else "cpu", enabled=cuda_amp):
            loss = fm_loss(model, target, observed, mask, hist)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        if step % args.log_every == 0 or step == args.steps:
            score = float(loss.detach())
            if valid is not None:
                model.eval()
                # Fixed validation randomness; retain training RNG state.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()] if cuda_amp else []):
                    torch.manual_seed(12345)
                    errors = []
                    std = torch.tensor(manifest["std"], device=device)
                    for i, (h, z) in enumerate(valid):
                        h, z = h.to(device), z.to(device)
                        m = random_mask(len(h), 5, args.drops, device)
                        rec = reconstruct(model, z*m[..., None], m, h, args.receiver_steps)
                        errors.append(float((((rec-z)*std).square()*(1-m[..., None])).sum() / ((1-m).sum()*z.shape[-1])))
                        if i+1 >= args.valid_batches:
                            break
                    score = float(np.mean(errors))
            improved = score < best
            best = min(best, score)
            row = {"step": step, "loss": float(loss.detach()), "score": score, "score_kind": "validation_missing_mse" if valid else "training_fm_only"}
            print(json.dumps(row), flush=True)
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row)+"\n")
            state = dict(model=model.state_dict(), model_config=config, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=step, best=best, manifest=manifest, args=vars(args))
            torch.save(state, out / "last.pt")
            if improved:
                torch.save(state, out / "best.pt")
