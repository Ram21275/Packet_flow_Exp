import csv
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .codec import SR, FRAME, decode_audio
from .data import load_manifest
from .models import random_mask, reconstruct, select
from .network import Channel
from .train import seed_all, load_model, assert_compatible, device_name


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def snr(reference, estimate):
    return float(10*np.log10((np.sum(reference**2)+1e-12)/(np.sum((reference-estimate)**2)+1e-12)))


def evaluate(args):
    seed_all(args.seed)
    device = device_name(args.device)
    manifest = load_manifest(args.data)
    receiver, ckpt = load_model(args.receiver, device)
    assert_compatible(ckpt, manifest)
    sender = None
    if "flow" in args.modes.split(","):
        if not args.sender:
            raise ValueError("--sender checkpoint required for flow mode")
        sender, sender_ckpt = load_model(args.sender, device)
        assert_compatible(sender_ckpt, manifest)
        if sender_ckpt["args"]["drops"] != args.drops:
            raise ValueError("Sender trained at a different omission budget")
    mean = torch.tensor(manifest["mean"], device=device)
    std = torch.tensor(manifest["std"], device=device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in args.modes.split(","):
        if mode not in {"full", "random", "energy", "flow"}:
            raise ValueError(mode)
        channel = Channel(args.loss, args.delay_ms, args.jitter_ms, args.bandwidth_kbps, args.deadline_ms, args.seed, args.burst)
        sender_ms, receiver_ms, scores = [], [], []
        total_frames, omitted, window_id = 0, 0, 0
        entries = manifest["splits"][args.split]
        for entry in entries[:args.eval_limit or None]:
            with np.load(Path(args.data) / entry["path"]) as archive:
                coeff = archive["z"].astype(np.float32)
            n = len(coeff)//5*5
            coeff = coeff[:n]
            z = (torch.tensor(coeff, device=device)-mean)/std
            rh = torch.zeros(1, 25, manifest["dim"], device=device)
            sh = torch.zeros_like(rh)  # sender's local shadow history, no loss feedback
            reconstructed = []
            for start in range(0, n, 5):
                block = z[start:start+5][None]
                local_window = start//5
                anchor = local_window % args.anchor_every == 0
                if anchor:
                    rh.zero_()
                    sh.zero_()
                # A per-window local seed synchronizes shadow/receiver generation;
                # shared session seed is modeled, not a transmitted per-frame secret.
                gen = torch.Generator(device=device).manual_seed(args.seed+window_id+100000)
                noise = torch.randn(block.shape, device=device, generator=gen)
                score_noise = torch.randn((1, 5), device=device, generator=gen)
                sync(device)
                begin = time.perf_counter()
                if anchor or mode == "full":
                    mask = torch.ones(1, 5, device=device)
                elif mode == "random":
                    mask = torch.ones(1, 5, device=device)
                    mask.scatter_(1, score_noise.argsort(-1)[:, :args.drops], 0.)
                elif mode == "energy":
                    mask = torch.ones(1, 5, device=device)
                    mask.scatter_(1, (block*std+mean).square().mean(-1).argsort(-1)[:, :args.drops], 0.)
                else:
                    mask = select(sender, block, sh, args.drops, args.sender_steps, score_noise)
                # Shadow reconstructs intended omissions only; never sees actual
                # network losses. This simulation has no receiver feedback.
                shadow = reconstruct(receiver, block*mask[..., None], mask, sh, args.receiver_steps, noise)
                sh = torch.cat([sh, shadow], 1)[:, -25:]
                sync(device)
                sender_ms.append((time.perf_counter()-begin)*1000)
                observed = torch.zeros_like(block)
                received = torch.zeros_like(mask)
                release = window_id*.1+.1  # full 100 ms collection before sending
                for pos in range(5):
                    if mask[0, pos] == 0:
                        omitted += 1
                        continue
                    # Receiver gets only bytes that survive channel + authentication.
                    frame = (block[0, pos]*std+mean).cpu().numpy()
                    decoded = channel.deliver(frame, window_id, pos, release)
                    if decoded is not None:
                        observed[0, pos] = (torch.tensor(decoded, device=device)-mean)/std
                        received[0, pos] = 1
                sync(device)
                begin = time.perf_counter()
                if args.concealment == "zero":
                    # Zero waveform corresponds to normalized coefficient -mean/std.
                    rec = observed + (1-received[..., None])*(-mean/std)
                else:
                    rec = reconstruct(receiver, observed, received, rh, args.receiver_steps, noise)
                sync(device)
                receiver_ms.append((time.perf_counter()-begin)*1000)
                rh = torch.cat([rh, rec], 1)[:, -25:]
                reconstructed.append((rec[0]*std+mean).cpu().numpy())
                total_frames += 5
                window_id += 1
            estimate = decode_audio(np.concatenate(reconstructed))
            reference, _ = sf.read(Path(args.data)/Path(entry["path"]).with_suffix(".wav"), dtype="float32")
            reference = reference[:n*FRAME]
            codec_reference = decode_audio(coeff)
            scores.append({"id": entry["id"], "snr_original_db": snr(reference, estimate), "snr_codec_db": snr(codec_reference, estimate), "codec_snr_db": snr(reference, codec_reference)})
            if len(scores) <= 3:
                sf.write(out/f"{mode}_{entry['id']}_reconstructed.wav", estimate, SR)
                sf.write(out/f"{mode}_{entry['id']}_original.wav", reference, SR)
                sf.write(out/f"{mode}_{entry['id']}_codec.wav", codec_reference, SR)
        duration = total_frames*.02
        row = dict(mode=mode, concealment=args.concealment, seed=args.seed, split=args.split,
                   loss=args.loss, burst=args.burst, audio_seconds=duration, total_frames=total_frames,
                   intentional_omissions=omitted, omission_fraction=omitted/max(total_frames, 1),
                   wire_kbps=channel.stats["wire_bytes"]*8/max(duration, 1e-9)/1000,
                   sender_p95_ms=float(np.percentile(sender_ms, 95)),
                   receiver_p95_ms=float(np.percentile(receiver_ms, 95)),
                   collection_ms=100, network_playout_budget_ms=args.deadline_ms,
                   snr_original_db=float(np.mean([s["snr_original_db"] for s in scores])),
                   snr_codec_db=float(np.mean([s["snr_codec_db"] for s in scores])),
                   **channel.stats)
        rows.append(row)
        (out/f"{mode}_utterances.json").write_text(json.dumps(scores, indent=2))
        print(json.dumps(row), flush=True)
    with (out/"summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/"config.json").write_text(json.dumps(vars(args), indent=2))
    print("SNR is diagnostic only; no WER/STOI/PESQ or mobile-performance claims.", flush=True)
