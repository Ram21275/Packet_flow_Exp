# Local validation — 2026-11-05

Executed on CPU in the development environment:

- Five pytest checks passed: codec distortion/payload size; sender budget and
  received-frame preservation with finite gradients; AEAD tamper rejection,
  exact modeled byte size and deadline erasure; paired independent/burst traces
  remain paired when intentional omissions differ; local extracted-FLAC split
  handling and training-only normalization.
- Synthetic 16-coefficient data generated with distinct train/valid/test audio.
- Receiver trained for three optimizer steps; validation and best/last saved.
- Eight teacher windows generated with repeated common-noise evaluation of all
  five one-omission masks and full expected-cost vectors.
- Expected-cost sender trained for three steps with held-out expected-regret
  checkpoint selection; deterministic checkpoint reload was verified.
- Full, random, energy and learned modes evaluated over a synthetic utterance;
  real encryption/decryption ran; CSV and listening examples were produced.
- Receiver checkpoint resume completed, and a burst-loss/low-bandwidth run
  exercised both channel losses and late-packet erasures.
- This verifies execution only. Three-step models are untrained for practical
  purposes, and synthetic audio cannot establish speech reconstruction quality.

Not yet established: useful LibriSpeech quality, Kaggle GPU behavior, advantage
over lower-bitrate Opus or a neural codec, real-time/mobile performance, perceptual
scores, end-to-end differentiable selection, or actual network-stack performance.

Public Hugging Face download was attempted here but was blocked by this
environment's proxy dependency/configuration before dataset transfer. The HF
route is implemented but must be exercised with Kaggle Internet enabled. The
local extracted-FLAC route is tested independently without internet.

On 2026-11-10, the revised teacher-to-selector-to-audit path also completed on a
tiny local cache. The selector remained invariant to score noise and nominal
integration-step arguments, and the audit emitted the learned-selector schema.
