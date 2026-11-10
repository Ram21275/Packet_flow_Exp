import itertools

import torch
from torch import nn


class ConditionalFlow(nn.Module):
    def __init__(self, dim=160, window=5, width=256, context=64, sender=False):
        super().__init__()
        self.dim, self.window, self.sender = dim, window, sender
        self.history = nn.GRU(dim, context, batch_first=True)
        out = window if sender else window * dim
        # state + observed/original content + mask + encoded history + time
        inp = out + window * dim + window + context + 3
        self.net = nn.Sequential(nn.Linear(inp, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, out))

    def forward(self, state, time, observed, mask, history):
        h = self.history(history)[1][-1]
        time = time.reshape(-1, 1).expand(len(state), 1)
        t = torch.cat([time, time.sin(), time.cos()], -1)
        inp = torch.cat([state.flatten(1), observed.flatten(1), mask.float(), h, t], -1)
        return self.net(inp).reshape_as(state)


class ExpectedCostSelector(nn.Module):
    """Predicts per-mask expected receiver cost as a correction to frame energy."""

    def __init__(self, dim=160, window=5, width=256, context=64, drops=1,
                 mean=None, std=None, sender=True):
        super().__init__()
        self.dim, self.window, self.drops, self.sender = dim, window, drops, True
        combinations, masks = omission_actions(window, drops)
        self.combinations = combinations
        self.register_buffer("action_masks", masks)
        self.register_buffer("coeff_mean", torch.tensor(mean if mean is not None else [0.] * dim))
        self.register_buffer("coeff_std", torch.tensor(std if std is not None else [1.] * dim))
        actions = len(combinations)
        self.history = nn.GRU(dim, context, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(window * dim + context + window + actions, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, actions),
        )
        # Before learning, selection exactly follows the physical-energy baseline.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def energy_scores(self, original):
        physical = original * self.coeff_std + self.coeff_mean
        frame_energy = physical.square().mean(dim=-1)
        scores = torch.stack([
            frame_energy[:, list(omitted)].sum(dim=1)
            for omitted in self.combinations
        ], dim=1)
        relative = scores - scores.min(dim=1, keepdim=True).values
        return frame_energy, relative / relative.mean(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, original, history):
        encoded_history = self.history(history)[1][-1]
        frame_energy, energy_scores = self.energy_scores(original)
        features = torch.cat(
            [original.flatten(1), encoded_history, frame_energy, energy_scores], dim=-1
        )
        return energy_scores + self.net(features)


def omission_actions(window, drops, device=None):
    combinations = list(itertools.combinations(range(window), drops))
    masks = torch.ones(len(combinations), window, device=device)
    for index, omitted in enumerate(combinations):
        masks[index, list(omitted)] = 0
    return combinations, masks


def random_mask(batch, window, drops, device):
    mask = torch.ones(batch, window, device=device)
    mask.scatter_(1, torch.rand(batch, window, device=device).argsort(-1)[:, :drops], 0.)
    return mask


def fm_loss(model, target, observed, mask, history):
    noise = torch.randn_like(target)
    t = torch.rand(len(target), device=target.device)
    shaped = t.view(-1, *([1] * (target.ndim-1)))
    state = noise * (1-shaped) + target * shaped
    if not model.sender:
        state = state * (1-mask[..., None]) + observed
    pred = model(state, t, observed, mask, history)
    error = (pred - (target-noise)).square()
    if model.sender:
        return error.mean()
    missing = 1-mask[..., None]
    return (error*missing).sum() / (missing.sum()*target.shape[-1]).clamp_min(1)


@torch.no_grad()
def reconstruct(model, observed, mask, history, steps=4, noise=None):
    if steps < 1:
        raise ValueError("steps must be positive")
    missing = 1-mask[..., None]
    state = (torch.randn_like(observed) if noise is None else noise.clone()) * missing + observed
    for i in range(steps):
        t = torch.full((len(state),), i/steps, device=state.device)
        state = state + model(state, t, observed, mask, history) * missing / steps
        state = state * missing + observed
    return state


@torch.no_grad()
def select(model, original, history, drops=1, steps=2, noise=None):
    if isinstance(model, ExpectedCostSelector):
        if drops != model.drops:
            raise ValueError("Selector omission budget differs from requested drops")
        action = model(original, history).argmin(dim=1)
        return model.action_masks[action]
    b, w, _ = original.shape
    state = torch.randn(b, w, device=original.device) if noise is None else noise.clone()
    budget = torch.full((b, w), 1-drops/w, device=original.device)
    for i in range(steps):
        state += model(state, torch.full((b,), i/steps, device=state.device), original, budget, history) / steps
    mask = torch.ones_like(state)
    mask.scatter_(1, state.argsort(-1)[:, :drops], 0.)
    return mask
