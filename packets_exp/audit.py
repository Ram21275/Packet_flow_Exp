import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .models import reconstruct, select


class DirectCostScorer(nn.Module):
    """Deterministic baseline that predicts one reconstruction cost per mask."""

    def __init__(self, dim, actions, window=5, width=256, context=64):
        super().__init__()
        self.config = dict(dim=dim, actions=actions, window=window, width=width, context=context)
        self.history = nn.GRU(dim, context, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(window * dim + context, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, actions),
        )

    def forward(self, original, history):
        h = self.history(history)[1][-1]
        return self.net(torch.cat([original.flatten(1), h], dim=-1))


def omission_actions(window, drops, device=None):
    combinations = list(itertools.combinations(range(window), drops))
    masks = torch.ones(len(combinations), window, device=device)
    for index, omitted in enumerate(combinations):
        masks[index, list(omitted)] = 0
    return combinations, masks


def masks_to_actions(masks, action_masks):
    matches = (masks[:, None, :] == action_masks[None, :, :]).all(dim=-1)
    if not matches.any(dim=1).all():
        raise ValueError("A selected mask does not match the configured omission budget")
    return matches.float().argmax(dim=1)


@torch.no_grad()
def reconstruction_cost_draws(receiver, original, history, std, action_masks,
                              noise_draws, receiver_steps):
    """Return [draw, batch, action] costs with common noise across actions."""
    all_draws = []
    for _ in range(noise_draws):
        common_noise = torch.randn_like(original)
        action_costs = []
        for action_mask in action_masks:
            mask = action_mask.expand(len(original), -1)
            rec = reconstruct(
                receiver,
                original * mask[..., None],
                mask,
                history,
                receiver_steps,
                common_noise,
            )
            action_costs.append(((rec - original) * std).square().mean((1, 2)))
        all_draws.append(torch.stack(action_costs, dim=1))
    return torch.stack(all_draws)


def teacher_stability(cost_draws):
    """Summarize how often repeated receiver draws choose the same teacher mask."""
    winners = cost_draws.argmin(dim=-1)  # draws, batch
    actions = cost_draws.shape[-1]
    probabilities = F.one_hot(winners, actions).float().mean(dim=0)
    modal_agreement = probabilities.max(dim=1).values
    entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=1)
    if actions > 1:
        entropy = entropy / np.log(actions)
    return {
        "modal_agreement": modal_agreement,
        "normalized_entropy": entropy,
        "unanimous": modal_agreement == 1,
        "modal_action": probabilities.argmax(dim=1),
    }


def policy_metrics(costs, choices):
    oracle = costs.argmin(dim=1)
    oracle_cost = costs.gather(1, oracle[:, None]).squeeze(1)
    selected_cost = costs.gather(1, choices[:, None]).squeeze(1)
    regret = selected_cost - oracle_cost
    return {
        "mean_cost": float(selected_cost.mean()),
        "oracle_mean_cost": float(oracle_cost.mean()),
        "mean_regret": float(regret.mean()),
        "median_regret": float(regret.median()),
        "p95_regret": float(torch.quantile(regret, .95)),
        "oracle_agreement": float((choices == oracle).float().mean()),
        "relative_regret": float(regret.mean() / oracle_cost.mean().clamp_min(1e-12)),
    }


def _energy_choices(original, mean, std, combinations):
    energy = (original * std + mean).square().mean(dim=-1)
    action_energy = torch.stack(
        [energy[:, list(omitted)].sum(dim=1) for omitted in combinations], dim=1
    )
    return action_energy.argmin(dim=1)


@torch.no_grad()
def _flow_choices(sender, original, history, action_masks, drops, steps, samples, seed):
    generator = torch.Generator(device=original.device).manual_seed(seed)
    choices = []
    for _ in range(samples):
        noise = torch.randn(original.shape[:2], device=original.device, generator=generator)
        mask = select(sender, original, history, drops, steps, noise)
        choices.append(masks_to_actions(mask, action_masks))
    return torch.stack(choices)


@torch.no_grad()
def _collect_examples(args, split, limit, receiver, manifest, device,
                      combinations, action_masks):
    from .data import Windows
    from .train import corrupt_history

    dataset = Windows(args.data, split)
    loader_generator = torch.Generator().manual_seed(args.seed + (0 if split == "train" else 1000))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split == "train",
        num_workers=0,
        generator=loader_generator,
    )
    mean = torch.tensor(manifest["mean"], device=device)
    std = torch.tensor(manifest["std"], device=device)
    saved = {"history": [], "original": [], "cost_draws": [], "energy": []}
    total = 0
    for history, original in loader:
        remaining = limit - total
        if remaining <= 0:
            break
        history = history[:remaining].to(device)
        original = original[:remaining].to(device)
        history = corrupt_history(
            receiver, history, args.rollout_fraction, args.receiver_steps
        )
        draws = reconstruction_cost_draws(
            receiver,
            original,
            history,
            std,
            action_masks,
            args.audit_noise_draws,
            args.receiver_steps,
        )
        saved["history"].append(history.cpu())
        saved["original"].append(original.cpu())
        saved["cost_draws"].append(draws.cpu())
        saved["energy"].append(
            _energy_choices(original, mean, std, combinations).cpu()
        )
        total += len(original)
        print(json.dumps({"audit_split": split, "windows": total, "target": limit}), flush=True)
    if total == 0:
        raise ValueError(f"No {split} windows available for sender audit")
    return {
        "history": torch.cat(saved["history"]),
        "original": torch.cat(saved["original"]),
        "cost_draws": torch.cat(saved["cost_draws"], dim=1),
        "energy": torch.cat(saved["energy"]),
    }


def _train_direct_baseline(args, train_examples, actions, device):
    model = DirectCostScorer(
        dim=train_examples["original"].shape[-1],
        actions=actions,
        width=args.width,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    costs = train_examples["cost_draws"].mean(dim=0)
    relative = costs - costs.min(dim=1, keepdim=True).values
    targets = relative / relative.mean(dim=1, keepdim=True).clamp_min(1e-8)
    oracle = costs.argmin(dim=1)
    count = len(costs)
    generator = torch.Generator().manual_seed(args.seed + 3000)
    last_loss = None
    model.train()
    for step in range(1, args.audit_baseline_steps + 1):
        indices = torch.randint(
            count, (min(args.batch_size, count),), generator=generator
        )
        original = train_examples["original"][indices].to(device)
        history = train_examples["history"][indices].to(device)
        target = targets[indices].to(device)
        labels = oracle[indices].to(device)
        predicted = model(original, history)
        loss = F.mse_loss(predicted, target) + .25 * F.cross_entropy(-predicted, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        last_loss = float(loss.detach())
        if step % max(args.audit_baseline_steps // 5, 1) == 0 or step == 1:
            print(json.dumps({"direct_baseline_step": step, "loss": last_loss}), flush=True)
    return model.eval(), last_loss


def _diagnostic_signals(policy, stability, integration, direct):
    selector_key = "learned" if "learned" in policy else "flow"
    selector_regret = policy[selector_key]["mean_regret"]
    energy_regret = policy["energy"]["mean_regret"]
    by_steps = {}
    for row in integration:
        by_steps.setdefault(row["steps"], []).append(row["mean_regret"])
    step_means = [sum(values) / len(values) for values in by_steps.values()]
    integration_range = max(step_means) - min(step_means)
    noise_range = max(max(values) - min(values) for values in by_steps.values())
    concern = {
        "teacher_unstable": stability["mean_modal_agreement"] < .80,
        "selector_worse_than_energy": policy[selector_key]["mean_cost"] > policy["energy"]["mean_cost"],
        "direct_baseline_beats_selector": direct["mean_regret"] < .90 * max(selector_regret, 1e-12),
        "integration_sensitive": integration_range > .20 * max(selector_regret, 1e-12),
        "selector_noise_sensitive": noise_range > .20 * max(selector_regret, 1e-12),
        "oracle_has_energy_headroom": energy_regret > .05 * max(policy["oracle"]["mean_cost"], 1e-12),
    }
    concern["formulation_concern_count"] = sum(bool(value) for value in concern.values())
    concern["thresholds_are_diagnostic_not_statistical_tests"] = True
    return concern


def audit_sender(args):
    from .data import load_manifest
    from .train import assert_compatible, device_name, load_model, seed_all

    if not args.sender:
        raise ValueError("--sender checkpoint is required for the sender audit")
    seed_all(args.seed)
    device = device_name(args.device)
    manifest = load_manifest(args.data)
    receiver, receiver_ckpt = load_model(args.receiver, device)
    sender, sender_ckpt = load_model(args.sender, device)
    assert_compatible(receiver_ckpt, manifest)
    assert_compatible(sender_ckpt, manifest)
    if not sender_ckpt["model_config"].get("sender"):
        raise ValueError("--sender checkpoint is not a sender model")
    if sender_ckpt["args"]["drops"] != args.drops:
        raise ValueError("Sender checkpoint omission budget differs from --drops")
    selector_key = (
        "learned"
        if sender_ckpt["model_config"].get("kind") == "expected_cost_selector"
        else "flow"
    )
    combinations, action_masks = omission_actions(5, args.drops, device)
    train_examples = _collect_examples(
        args,
        "train",
        args.audit_train_windows,
        receiver,
        manifest,
        device,
        combinations,
        action_masks,
    )
    valid_examples = _collect_examples(
        args,
        "valid",
        args.audit_valid_windows,
        receiver,
        manifest,
        device,
        combinations,
        action_masks,
    )
    valid_costs = valid_examples["cost_draws"].mean(dim=0)
    oracle = valid_costs.argmin(dim=1)
    random_generator = torch.Generator().manual_seed(args.seed + 4000)
    random_choices = torch.randint(
        len(combinations), (len(valid_costs),), generator=random_generator
    )
    original = valid_examples["original"].to(device)
    history = valid_examples["history"].to(device)
    flow_samples = _flow_choices(
        sender,
        original,
        history,
        action_masks,
        args.drops,
        args.sender_steps,
        args.audit_score_samples,
        args.seed + 5000,
    ).cpu()
    flow_choices = flow_samples[0]
    policy = {
        "oracle": policy_metrics(valid_costs, oracle),
        "random": policy_metrics(valid_costs, random_choices),
        "energy": policy_metrics(valid_costs, valid_examples["energy"]),
        selector_key: policy_metrics(valid_costs, flow_choices),
    }
    stability_values = teacher_stability(valid_examples["cost_draws"])
    stability = {
        "noise_draws": args.audit_noise_draws,
        "mean_modal_agreement": float(stability_values["modal_agreement"].mean()),
        "unanimous_fraction": float(stability_values["unanimous"].float().mean()),
        "mean_normalized_entropy": float(stability_values["normalized_entropy"].mean()),
    }
    integration = []
    for steps in args.audit_integration_steps:
        choices = _flow_choices(
            sender,
            original,
            history,
            action_masks,
            args.drops,
            steps,
            args.audit_score_samples,
            args.seed + 6000 + steps,
        ).cpu()
        for sample_index, sample_choices in enumerate(choices):
            row = {"steps": steps, "noise_sample": sample_index}
            row.update(policy_metrics(valid_costs, sample_choices))
            integration.append(row)
    direct_model, direct_train_loss = _train_direct_baseline(
        args, train_examples, len(combinations), device
    )
    with torch.no_grad():
        direct_choices = direct_model(original, history).argmin(dim=1).cpu()
    direct = policy_metrics(valid_costs, direct_choices)
    direct["training_loss"] = direct_train_loss
    direct["training_windows"] = len(train_examples["original"])
    signals = _diagnostic_signals(policy, stability, integration, direct)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "seed": args.seed,
            "drops": args.drops,
            "receiver_steps": args.receiver_steps,
            "sender_steps": args.sender_steps,
            "noise_draws": args.audit_noise_draws,
            "score_samples": args.audit_score_samples,
            "train_windows": len(train_examples["original"]),
            "valid_windows": len(valid_examples["original"]),
            "integration_steps": args.audit_integration_steps,
            "selector_kind": sender_ckpt["model_config"].get("kind", "conditional_flow"),
        },
        "test_1_oracle_benchmark": policy,
        "test_2_teacher_stability": stability,
        "test_3_sender_regret": policy[selector_key],
        "test_4_integration_ablation": integration,
        "test_5_direct_cost_baseline": direct,
        "diagnostic_signals": signals,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "integration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(integration[0]))
        writer.writeheader()
        writer.writerows(integration)
    with (output / "windows.csv").open("w", newline="") as handle:
        fields = [
            "window", "oracle_action", "random_action", "energy_action", "selector_action",
            "direct_action", "teacher_modal_agreement", "teacher_normalized_entropy",
        ] + [f"cost_action_{index}" for index in range(len(combinations))]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(valid_costs)):
            row = {
                "window": index,
                "oracle_action": int(oracle[index]),
                "random_action": int(random_choices[index]),
                "energy_action": int(valid_examples["energy"][index]),
                "selector_action": int(flow_choices[index]),
                "direct_action": int(direct_choices[index]),
                "teacher_modal_agreement": float(stability_values["modal_agreement"][index]),
                "teacher_normalized_entropy": float(stability_values["normalized_entropy"][index]),
            }
            row.update({
                f"cost_action_{action}": float(valid_costs[index, action])
                for action in range(len(combinations))
            })
            writer.writerow(row)
    torch.save(
        {"model": direct_model.state_dict(), "model_config": direct_model.config, "manifest": manifest},
        output / "direct_cost_scorer.pt",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Sender audit written to {output}", flush=True)
    return summary
