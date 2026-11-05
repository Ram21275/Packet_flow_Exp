import numpy as np
import pytest
import torch
from cryptography.exceptions import InvalidTag
from packets_exp.codec import encode_audio, decode_audio, pack_coeff, unpack_coeff
from packets_exp.models import ConditionalFlow, reconstruct, select, fm_loss
from packets_exp.network import Channel
from packets_exp.data import prepare, load_manifest
import soundfile as sf


def test_codec_full_band_and_payload():
    rng = np.random.default_rng(1)
    x = rng.normal(0, .1, 320*5).astype(np.float32)
    z = encode_audio(x, 320)
    y = decode_audio(z)
    assert np.mean((x-y)**2) < 2e-6
    payload = pack_coeff(z[0])
    assert len(payload) == 322
    assert np.max(np.abs(unpack_coeff(payload, 320)-z[0])) < .003


def test_flow_preserves_received_and_budget():
    torch.manual_seed(1)
    rx = ConditionalFlow(dim=8, width=16)
    tx = ConditionalFlow(dim=8, width=16, sender=True)
    z, h = torch.randn(2, 5, 8), torch.randn(2, 25, 8)
    m = select(tx, z, h, drops=1)
    assert torch.all(m.sum(-1) == 4)
    observed = z*m[..., None]
    restored = reconstruct(rx, observed, m, h)
    assert torch.equal(restored[m.bool()], z[m.bool()])
    loss = fm_loss(rx, z, observed, m, h)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in rx.parameters())


def test_wire_bytes_authentication_and_deadline():
    ch = Channel(delay_ms=0, jitter_ms=0, bandwidth_kbps=1000)
    coeff = np.arange(8, dtype=np.float32)*.01
    decoded = ch.deliver(coeff, 0, 0, .1)
    assert decoded is not None
    assert ch.stats['wire_bytes'] == (8+2)+5+12+16+28
    nonce, aad = b'\0'*12, b'header'
    enc = ch.aead.encrypt(nonce, b'content', aad)
    tampered = enc[:-1]+bytes([enc[-1]^1])
    with pytest.raises(InvalidTag):
        ch.aead.decrypt(nonce, tampered, aad)
    late = Channel(delay_ms=200, deadline_ms=100)
    assert late.deliver(coeff, 0, 0, .1) is None
    assert late.stats['late'] == 1


def test_paired_loss_trace_survives_omissions():
    for burst in (False, True):
        a = Channel(loss=.3, burst=burst, seed=19, jitter_ms=0, bandwidth_kbps=1e6)
        b = Channel(loss=.3, burst=burst, seed=19, jitter_ms=0, bandwidth_kbps=1e6)
        for slot in range(40):
            x = a.deliver(np.ones(8), slot//5, slot%5, slot*.1)
            if slot%3:
                y = b.deliver(np.ones(8), slot//5, slot%5, slot*.1)
                assert (x is None) == (y is None)


def test_local_dataset_splits_and_training_only_normalization(tmp_path):
    source = tmp_path/'LibriSpeech'
    rng = np.random.default_rng(13)
    for name, amplitude in [('train-clean-100', .05), ('dev-clean', .2), ('test-clean', .3)]:
        folder = source/name/'speaker'/'chapter'
        folder.mkdir(parents=True)
        sf.write(folder/'sample.flac', rng.normal(0, amplitude, 320*40), 16000)
    cache = tmp_path/'cache'
    prepare(cache, 1, 1, 1, 8, source)
    manifest = load_manifest(cache)
    with np.load(cache/manifest['splits']['train'][0]['path']) as data:
        assert np.allclose(manifest['mean'], data['z'].mean(0), atol=1e-7)
    assert set(manifest['splits']) == {'train', 'valid', 'test'}
    with pytest.raises(FileExistsError):
        prepare(cache, 1, 1, 1, 8, source)
