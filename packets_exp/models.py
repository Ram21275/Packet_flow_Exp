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
    b, w, _ = original.shape
    state = torch.randn(b, w, device=original.device) if noise is None else noise.clone()
    budget = torch.full((b, w), 1-drops/w, device=original.device)
    for i in range(steps):
        state += model(state, torch.full((b,), i/steps, device=state.device), original, budget, history) / steps
    mask = torch.ones_like(state)
    mask.scatter_(1, state.argsort(-1)[:, :drops], 0.)
    return mask
