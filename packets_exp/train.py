import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import Windows, load_manifest
from .models import (
    ConditionalFlow,
    ExpectedCostSelector,
    fm_loss,
    omission_actions,
    random_mask,
    reconstruct,
)


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
    config = dict(ckpt["model_config"])
    kind = config.pop("kind", "conditional_flow")
    if kind == "expected_cost_selector":
        model_class = ExpectedCostSelector
    elif kind == "conditional_flow":
        model_class = ConditionalFlow
    else:
        raise ValueError(f"Unknown checkpoint model kind: {kind}")
    model = model_class(**config).to(device)
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


@torch.no_grad()
def expected_action_costs(receiver, original, history, std, action_masks,
                          noise_draws, receiver_steps):
    """Estimate every action with repeated draws and common noise per draw."""
    draws = []
    for _ in range(noise_draws):
        common_noise = torch.randn_like(original)
        costs = []
        for action_mask in action_masks:
            mask = action_mask.expand(len(original), -1)
            reconstruction = reconstruct(
                receiver,
                original * mask[..., None],
                mask,
                history,
                receiver_steps,
                common_noise,
            )
            costs.append(((reconstruction - original) * std).square().mean((1, 2)))
        draws.append(torch.stack(costs, dim=1))
    draws = torch.stack(draws)
    return draws.mean(dim=0), draws.std(dim=0, unbiased=False)


def normalize_action_costs(costs):
    relative = costs - costs.min(dim=1, keepdim=True).values
    return relative / relative.mean(dim=1, keepdim=True).clamp_min(1e-8)


def selector_loss(predicted, costs, temperature=.5, rank_weight=.25):
    """Regress expected regret while treating near-tied actions as soft targets."""
    target = normalize_action_costs(costs)
    soft_target = torch.softmax(-target / temperature, dim=1)
    regression = torch.nn.functional.smooth_l1_loss(predicted, target)
    ranking = -(soft_target * torch.log_softmax(-predicted / temperature, dim=1)).sum(1).mean()
    return regression + rank_weight * ranking


@torch.no_grad()
def selector_validation_regret(model, loader, device):
    model.eval()
    regrets = []
    for history, original, action_costs in loader:
        history = history.to(device)
        original = original.to(device)
        action_costs = action_costs.to(device)
        choice = model(original, history).argmin(dim=1)
        selected = action_costs.gather(1, choice[:, None]).squeeze(1)
        regrets.append((selected - action_costs.min(dim=1).values).mean().item())
    return float(np.mean(regrets))


def teacher(args):
    seed_all(args.seed)
    device = device_name(args.device)
    receiver, ckpt = load_model(args.receiver, device)
    manifest = load_manifest(args.data)
    assert_compatible(ckpt, manifest)
    dataset = Windows(args.data, "train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    combinations, action_masks = omission_actions(5, args.drops, device)
    saved = {"history": [], "original": [], "costs": [], "cost_std": []}
    # Distortion measured in unnormalized coefficient space (orthonormal DCT).
    std = torch.tensor(manifest["std"], device=device)
    total = 0
    for hist, original in loader:
        hist, original = hist.to(device), original.to(device)
        hist = corrupt_history(receiver, hist, args.rollout_fraction, args.receiver_steps)
        costs, cost_std = expected_action_costs(
            receiver,
            original,
            hist,
            std,
            action_masks,
            args.teacher_noise_draws,
            args.receiver_steps,
        )
        saved["history"].append(hist.cpu())
        saved["original"].append(original.cpu())
        saved["costs"].append(costs.cpu())
        saved["cost_std"].append(cost_std.cpu())
        total += len(hist)
        print(json.dumps({"teacher_windows": total, "target": args.teacher_windows}), flush=True)
        if total >= args.teacher_windows:
            break
    out = {k: torch.cat(v)[:args.teacher_windows] for k, v in saved.items()}
    out.update(
        manifest=manifest,
        drops=args.drops,
        receiver=str(args.receiver),
        seed=args.seed,
        format="expected_cost_v2",
        noise_draws=args.teacher_noise_draws,
        combinations=combinations,
        action_masks=action_masks.cpu(),
    )
    path = Path(args.teacher)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"Saved {len(out['costs'])} expected-cost teacher windows to {path}", flush=True)


def train(args):
    seed_all(args.seed)
    device = device_name(args.device)
    manifest = load_manifest(args.data)
    if args.stage == "sender":
        config = dict(
            kind="expected_cost_selector",
            dim=manifest["dim"],
            window=5,
            width=args.width,
            context=64,
            drops=args.drops,
            mean=manifest["mean"],
            std=manifest["std"],
            sender=True,
        )
        model = ExpectedCostSelector(**{k: v for k, v in config.items() if k != "kind"}).to(device)
    else:
        config = dict(dim=manifest["dim"], window=5, width=args.width, context=64, sender=False)
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
        if cache.get("format") != "expected_cost_v2" or "costs" not in cache:
            raise ValueError("Legacy hard-label teacher cache; regenerate it with the teacher command")
        full_dataset = torch.utils.data.TensorDataset(
            cache["history"], cache["original"], cache["costs"]
        )
        valid_count = max(1, round(len(full_dataset) * args.sender_valid_fraction))
        train_count = len(full_dataset) - valid_count
        if train_count < 1:
            raise ValueError("Teacher cache needs at least two windows")
        dataset, valid_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_count, valid_count],
            generator=torch.Generator().manual_seed(args.seed + 2000),
        )
        valid = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0)
    else:
        dataset = Windows(args.data, "train")
        valid = DataLoader(Windows(args.data, "valid"), batch_size=args.batch_size, num_workers=0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    iterator = iter(loader)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(json.dumps({"stage": args.stage, "device": device, "parameters": sum(p.numel() for p in model.parameters()), "windows": len(dataset)}), flush=True)
    if args.stage == "sender" and start == 0:
        best = selector_validation_regret(model, valid, device)
        initial_state = dict(
            model=model.state_dict(),
            model_config=config,
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            step=0,
            best=best,
            manifest=manifest,
            args=vars(args),
        )
        torch.save(initial_state, out / "best.pt")
        print(json.dumps({
            "step": 0,
            "loss": None,
            "score": best,
            "score_kind": "validation_expected_regret_energy_initialization",
        }), flush=True)
    for step in range(start+1, args.steps+1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model.train()
        if args.stage == "sender":
            hist, original, costs = (x.to(device) for x in batch)
        else:
            hist, target = (x.to(device) for x in batch)
            hist = corrupt_history(model, hist, args.rollout_fraction, args.receiver_steps)
            # Mixture of intentional omissions and additional random channel loss.
            mask = random_mask(len(hist), 5, args.drops, device)
            mask *= (torch.rand_like(mask) > args.train_loss).float()
            observed = target * mask[..., None]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if cuda_amp else "cpu", enabled=cuda_amp):
            if args.stage == "sender":
                loss = selector_loss(
                    model(original, hist),
                    costs,
                    args.sender_temperature,
                    args.sender_rank_weight,
                )
            else:
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
            if args.stage == "sender":
                score = selector_validation_regret(model, valid, device)
            elif valid is not None:
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
            score_kind = "validation_expected_regret" if args.stage == "sender" else "validation_missing_mse"
            row = {"step": step, "loss": float(loss.detach()), "score": score, "score_kind": score_kind}
            print(json.dumps(row), flush=True)
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row)+"\n")
            state = dict(model=model.state_dict(), model_config=config, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=step, best=best, manifest=manifest, args=vars(args))
            torch.save(state, out / "last.pt")
            if improved:
                torch.save(state, out / "best.pt")
