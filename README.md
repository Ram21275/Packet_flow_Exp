# Dual-flow packet omission experiments

Simulation-first research prototype: a sender conditional flow selects which
application frames to omit; a receiver conditional flow reconstructs their content.
This is **lossy speech reconstruction**, not exact recovery of arbitrary bytes or
encrypted network packets. There are no trained results or hardware claims yet.

## Implemented milestone

- 16 kHz speech, 20 ms frames, five-frame (100 ms) selection windows.
- Frozen reference DCT codec: 160 of 320 orthonormal coefficients, symmetric int8
  quantization and float16 per-frame scale. Each frame is independently decodable.
  At 160 coefficients this is 64.8 kb/s payload before headers, NOT a competitive
  speech codec. Changing `--dim` changes the reference rate and bandwidth limit.
- Receiver conditional flow matching and a small GRU-conditioned velocity MLP.
- Exact enumeration of the five one-omission masks for training-only teacher
  targets; immediate coefficient distortion scoring, not future-horizon optimality.
- Sender continuous flow on dequantized mask scores, followed by hard selection.
- Receiver history exposure from one self-generated preceding window during
  training; full utterance rollouts during evaluation. Training is not yet a full
  long-horizon joint optimization or an adaptive rate controller.
- Real AES-GCM encryption/decryption of serialized per-frame payloads; modeled
  IPv4+UDP overhead; bandwidth queue, Gaussian jitter, independent/Gilbert loss,
  deadlines and scheduled full-transmission anchors every ten windows.
- No receiver loss feedback: sender uses its own shadow receiver and can diverge
  under actual loss. Both sides reset history at scheduled anchor windows, even
  when some anchor packets are lost. No implicit reliable delivery is assumed.
- CSV metrics, checkpoint resume, WAV examples, and synthetic smoke tests.

## Kaggle setup

Select a GPU accelerator and enable Internet in notebook settings. Keep Kaggle's
installed PyTorch rather than reinstalling it. GPU quota and session duration
depend on your account. The pilot does not require access to any private model.

```python
!git clone https://github.com/Ram21275/Packet_flow_Exp.git /kaggle/working/dual-flow-packets
%cd /kaggle/working/dual-flow-packets
%pip install -q -e ".[test]"
!python -m pytest -q
```

Run the pilot in the next cell:

```python
!bash scripts/pilot.sh /kaggle/working/packets_data /kaggle/working/packets_outputs
```

The pilot downloads **1000 training utterances, not 1000 hours or 10 hours**;
actual duration depends on utterance lengths. 100 validation and 100 test
utterances are cached separately. Selection uses validation by default; test
evaluation is an explicit later command. These are smoke/pilot settings, not a
publication-scale training prescription. Adjust training steps after inspecting
learning curves and listening to examples. No runtime estimate is guaranteed.

If a run is interrupted, preserve `/kaggle/working/packets_outputs` through Kaggle
notebook output saving. To resume receiver optimization to a total of 10000 steps:

```python
!python -m packets_exp.cli train --stage receiver --data /kaggle/working/packets_data --out /kaggle/working/packets_outputs/receiver --resume /kaggle/working/packets_outputs/receiver/last.pt --steps 10000
```

Resume restores optimizer/scaler/step but restarts the seeded DataLoader shuffle;
it is not bitwise replay. Only load checkpoints from trusted sources.
Rerunning pilot.sh starts a new training run; use explicit resume commands after
interruptions. Use new output directories for independent experiments.

## Dataset: Hugging Face or extracted Kaggle data

Official dataset mirror: https://huggingface.co/datasets/openslr/librispeech_asr
Original dataset: https://www.openslr.org/12/

The default `prepare` command streams Hugging Face's `clean` configuration:
`train.100` -> train, `validation` -> valid, `test` -> test. It uses
`Audio(decode=False)` and SoundFile on embedded FLAC bytes, so it does not require
TorchCodec/FFmpeg. No HF token is required for this public dataset. Internet is
required unless the data is already attached to the notebook.

```python
!python -m packets_exp.cli prepare --data /kaggle/working/packets_data --train-limit 1000 --valid-limit 100 --test-limit 100
```

If you attach a Kaggle copy, locate its extracted `LibriSpeech` directory. Pass
that exact root; it must contain `train-clean-100/`, `dev-clean/`, and `test-clean/`:

```python
!python -m packets_exp.cli prepare --data /kaggle/working/packets_data --local-root /kaggle/input/YOUR-DATASET/LibriSpeech --train-limit 1000 --valid-limit 100 --test-limit 100
```

The local route preserves official splits. Do not randomly divide audio windows
from the same utterance into train/test. The manifest stores normalization fitted
on training only. All checkpoints are checked against the cache manifest.
Use a **new cache directory** to change codec dimensions or data limits.

For full train-clean-100, use `--train-limit 0 --valid-limit 0 --test-limit 0`.
Check disk/RAM first: original WAVs and float32 latent caches can use tens of GB;
the current window dataset loads one split's latents into RAM. Start with pilot
limits on Kaggle. Cache preparation can be rerun after a partial failed download;
it refuses to overwrite an already completed manifest.

## Individual commands and experiment order

```bash
python -m packets_exp.cli train --stage receiver --data data/pilot --out outputs/receiver --steps 5000
python -m packets_exp.cli teacher --data data/pilot --receiver outputs/receiver/best.pt --teacher outputs/teacher.pt --teacher-windows 20000
python -m packets_exp.cli train --stage sender --data data/pilot --teacher outputs/teacher.pt --out outputs/sender --steps 3000
python -m packets_exp.cli evaluate --data data/pilot --receiver outputs/receiver/best.pt --sender outputs/sender/last.pt --out outputs/valid_clean
```

Receiver `best.pt` uses fixed-mask validation missing-coefficient MSE. Sender
`best.pt` is based on training FM loss only, so evaluation commands deliberately
use sender `last.pt`. Choose sender training duration using separate validation
rollouts, not the test set.

Compare `full`, `random`, `energy`, and `flow` under the same slot-indexed channel
trace. Full transmission is the codec reference; energy is a simple sender
baseline. `--concealment zero` gives a nonlearned receiver baseline. All modes
currently maintain a shadow receiver, so sender time includes that compute;
baseline sender time is not an optimized conventional codec implementation.

Evaluate seeds 7, 8, and 9; loss 0, .01, .03, .05; both independent and `--burst`;
flow integration steps 1, 2, and 4. Train independent checkpoints at multiple
seeds for model-variance claims. Channel-only seeds are insufficient for that.
After validation choices are frozen:

```bash
python -m packets_exp.cli evaluate --data data/pilot --receiver outputs/receiver/best.pt --sender outputs/sender/last.pt --split test --eval-limit 0 --out outputs/test_clean
```

## Sender formulation audit

Before extending sender training or changing the flow, run the five-part audit on
the validation split. It uses fresh train windows for the direct baseline and
fresh validation windows for all comparisons; the test split remains untouched.

```bash
python -m packets_exp.cli audit \
  --data /kaggle/working/packets_data_360 \
  --receiver /kaggle/working/packets_outputs/receiver/best.pt \
  --sender /kaggle/working/packets_outputs/sender/last.pt \
  --out /kaggle/working/packets_outputs/sender_audit \
  --audit-train-windows 2000 \
  --audit-valid-windows 500 \
  --audit-noise-draws 4 \
  --audit-score-samples 4 \
  --audit-baseline-steps 1000 \
  --audit-integration-steps 1,2,4,8
```

The command runs:

1. an oracle-mask benchmark against random, energy and flow selection;
2. teacher-label stability across repeated receiver noise draws;
3. sender regret relative to the per-window oracle;
4. sender integration-step and initial-noise ablations; and
5. a deterministic network that directly predicts all mask costs.

Results are written to `summary.json`, `windows.csv`, `integration.csv`, and
`direct_cost_scorer.pt`. `diagnostic_signals` in the summary applies documented
heuristic thresholds; these are formulation warnings, not statistical hypothesis
tests. Evidence against the current flow is strongest when the oracle has useful
headroom, the direct scorer has substantially lower regret, and the flow result is
sensitive to integration steps or initial noise. If the oracle barely improves on
energy, changing the sender model alone is unlikely to help.

`--drops 2` implements 40% omission in non-anchor windows. It is for stress tests,
allows adjacent omissions across boundaries, and does not enforce a global
maximum gap. Default `--drops 1` gives 20% omission in non-anchor windows, with
lower actual omission because anchors are sent in full. Train sender/teacher at
the same omission budget. A 10% target needs a different schedule/window design.

## Protocol and measurement boundaries

Per sent frame: 5-byte authenticated header (window ID, position), 12-byte nonce,
16-byte GCM tag, D+2 codec bytes, plus 28 modeled IPv4/UDP bytes. At D=160 this
is 223 bytes per frame. Nonces are unique within a randomly keyed simulation
session. No keys are committed or written to reports.

No separate omission mask is sent: both endpoints know the frame clock/window
schedule, and absent IDs at the playback deadline are erasures whether deliberate
or accidental. Thus the receiver never receives a loss-free side channel telling
it original omitted content. Model version, codec config, clock and shared random
seed are assumed pre-negotiated; setup/handshake bytes are currently excluded.
This is a UDP abstraction, not an implementation or benchmark of QUIC, TCP,
congestion control, real sockets, packet fragmentation, retransmission or radio
energy. Deployment needs authenticated session negotiation and replay protection.

Sender collects 100 ms; receiver uses a configurable network deadline after
release. `sender_p95_ms` and `receiver_p95_ms` are measured wall-clock inference,
not fed into the simulated clock. Codec and serialization runtime are excluded.
Collection and deadline delay are reported separately. This simulator cannot
prove real-time schedulability or mobile power consumption.

`summary.csv` reports actual transmitted serialized bytes plus modeled headers,
network drops, late frames, omission rate, SNR vs original and vs codec reference.
SNR is only a diagnostic; listen to WAVs. WER/STOI/PESQ, perceptual confidence
intervals, learned deterministic selector ablation, matched-rate Opus and neural
codec comparisons, residual corrections, joint fine-tuning and mobile profiling
remain required extensions. No bandwidth advantage over modern codecs is claimed.

## Synthetic end-to-end check (no dataset download)

```bash
python -m packets_exp.cli synthetic --data /tmp/flow-data --dim 16
python -m packets_exp.cli train --data /tmp/flow-data --out /tmp/flow-rx --steps 3 --log-every 1 --width 32 --batch-size 4 --valid-batches 1
python -m packets_exp.cli teacher --data /tmp/flow-data --receiver /tmp/flow-rx/best.pt --teacher /tmp/flow-teacher.pt --teacher-windows 8 --batch-size 4
python -m packets_exp.cli train --stage sender --data /tmp/flow-data --teacher /tmp/flow-teacher.pt --out /tmp/flow-tx --steps 3 --log-every 1 --width 32 --batch-size 4
python -m packets_exp.cli evaluate --data /tmp/flow-data --receiver /tmp/flow-rx/best.pt --sender /tmp/flow-tx/last.pt --eval-limit 1 --out /tmp/flow-eval
```

This checks execution and invariants, not reconstruction quality. Run `python -m
pytest -q` for codec, budget, observed-frame preservation, authentication,
deadline and paired-channel tests.

## References and data rights

- LibriSpeech: Panayotov et al., ICASSP 2015, https://www.openslr.org/12/
- Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- Flow-PLC (existing flow-based speech concealment): DOI 10.1109/LSP.2025.3553421

Code is MIT-licensed. Dataset rights are separate; consult the dataset card.
Do not commit dataset audio, private recordings, tokens or model checkpoints.
