import argparse
import torch


def main():
    p = argparse.ArgumentParser(description="Dual conditional-flow packet omission prototype")
    p.add_argument("command", choices=["prepare", "synthetic", "train", "teacher", "evaluate"])
    p.add_argument("--data", default="data/pilot")
    p.add_argument("--out", default="outputs/run")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--dim", type=int, default=160)
    p.add_argument("--train-limit", type=int, default=1000, help="utterances; 0 means all")
    p.add_argument("--valid-limit", type=int, default=100)
    p.add_argument("--test-limit", type=int, default=100)
    p.add_argument("--local-root", help="extracted LibriSpeech root with train-clean-100/dev-clean/test-clean")
    p.add_argument("--stage", choices=["receiver", "sender"], default="receiver")
    p.add_argument("--steps", type=int, default=5000, help="total optimizer steps, including resumed steps")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--valid-batches", type=int, default=8)
    p.add_argument("--receiver", default="outputs/receiver/best.pt")
    p.add_argument("--sender", default=None)
    p.add_argument("--resume")
    p.add_argument("--teacher", default="outputs/teacher.pt")
    p.add_argument("--teacher-windows", type=int, default=20000)
    p.add_argument("--receiver-steps", type=int, default=4)
    p.add_argument("--sender-steps", type=int, default=2)
    p.add_argument("--drops", type=int, default=1)
    p.add_argument("--train-loss", type=float, default=.02)
    p.add_argument("--rollout-fraction", type=float, default=.25)
    p.add_argument("--modes", default="full,random,energy,flow")
    p.add_argument("--concealment", choices=["flow", "zero"], default="flow")
    p.add_argument("--split", choices=["valid", "test"], default="valid")
    p.add_argument("--eval-limit", type=int, default=20, help="0 means all")
    p.add_argument("--loss", type=float, default=0.)
    p.add_argument("--burst", action="store_true")
    p.add_argument("--delay-ms", type=float, default=20.)
    p.add_argument("--jitter-ms", type=float, default=5.)
    p.add_argument("--deadline-ms", type=float, default=100.)
    p.add_argument("--bandwidth-kbps", type=float, default=256.)
    p.add_argument("--anchor-every", type=int, default=10)
    a = p.parse_args()
    if not 1 <= a.drops <= 2:
        p.error("initial experiment supports --drops 1 or 2")
    if a.anchor_every < 1 or a.batch_size < 1 or a.steps < 1 or a.log_every < 1:
        p.error("anchor-every, batch-size, steps and log-every must be positive")
    if not 0 <= a.rollout_fraction <= 1 or not 0 <= a.train_loss < 1:
        p.error("invalid probability")
    if any(n < 0 for n in [a.train_limit, a.valid_limit, a.test_limit, a.eval_limit]):
        p.error("limits must be nonnegative")
    torch.set_num_threads(a.threads)
    if a.command == "prepare":
        from .data import prepare
        prepare(a.data, a.train_limit, a.valid_limit, a.test_limit, a.dim, a.local_root)
    elif a.command == "synthetic":
        from .data import synthetic
        synthetic(a.data, a.dim)
    elif a.command in ("train", "teacher"):
        from .train import train, teacher
        (train if a.command == "train" else teacher)(a)
    else:
        from .evaluate import evaluate
        evaluate(a)


if __name__ == "__main__":
    main()
