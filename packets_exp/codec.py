"""Simple reference transform codec, NOT an Opus/neural codec replacement.

20 ms non-overlapping frames, orthonormal DCT, first D coefficients,
per-frame symmetric int8 quantization and float16 scale. Payload D+2 bytes.
"""
import numpy as np
from scipy.fft import dct, idct

SR = 16000
FRAME = 320


def pack_coeff(coeff):
    coeff = np.asarray(coeff, dtype=np.float32)
    scale = np.float16(max(float(np.abs(coeff).max()) / 127, 1e-6))
    q = np.clip(np.rint(coeff / float(scale)), -127, 127).astype(np.int8)
    return np.asarray(scale, dtype="<f2").tobytes() + q.tobytes()


def unpack_coeff(payload, dim):
    if len(payload) != dim + 2:
        raise ValueError("Invalid codec payload length")
    scale = float(np.frombuffer(payload[:2], dtype="<f2")[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid codec scale")
    return np.frombuffer(payload[2:], dtype=np.int8).astype(np.float32) * scale


def encode_audio(audio, dim=160):
    if not 1 <= dim <= FRAME:
        raise ValueError("dim must be in [1, 320]")
    n = len(audio) // FRAME
    coeff = dct(np.asarray(audio[:n * FRAME]).reshape(n, FRAME), norm="ortho", axis=-1)[:, :dim]
    return np.stack([unpack_coeff(pack_coeff(z), dim) for z in coeff])


def decode_audio(coeff):
    padded = np.zeros((len(coeff), FRAME), dtype=np.float32)
    padded[:, :coeff.shape[-1]] = coeff
    return idct(padded, norm="ortho", axis=-1).reshape(-1)
