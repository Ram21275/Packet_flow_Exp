"""AEAD datagram simulation with serialization, queue delay, jitter and loss.

Not QUIC: counts IPv4+UDP overhead, with no Ethernet or congestion controller.
Each frame has explicit window/position IDs; missing IDs are deadline erasures.
"""
import struct
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .codec import pack_coeff, unpack_coeff


class Channel:
    def __init__(self, loss=0., delay_ms=20., jitter_ms=5., bandwidth_kbps=256., deadline_ms=100., seed=1, burst=False):
        if not 0 <= loss < 1 or bandwidth_kbps <= 0 or deadline_ms < 0 or jitter_ms < 0:
            raise ValueError("Invalid network configuration")
        self.loss, self.delay, self.jitter = loss, delay_ms/1000, jitter_ms/1000
        self.rate, self.deadline = bandwidth_kbps*1000/8, deadline_ms/1000
        self.rng = np.random.default_rng(seed)
        self.seed, self.last_slot = seed, -1
        self.key = AESGCM.generate_key(bit_length=128)
        self.aead = AESGCM(self.key)
        self.counter, self.queue_free = 0, 0.
        self.burst, self.bad = burst, False
        self.stats = dict(wire_bytes=0, payload_bytes=0, sent_frames=0, network_lost=0, late=0, auth_failed=0)

    def deliver(self, coeff, window_id, position, release):
        payload = pack_coeff(coeff)
        header = struct.pack("!IB", window_id, position)
        nonce = self.counter.to_bytes(12, "big")
        self.counter += 1
        ciphertext = self.aead.encrypt(nonce, payload, header)
        wire = header + nonce + ciphertext
        size = len(wire) + 28  # modeled IPv4 (20) and UDP (8)
        self.stats["wire_bytes"] += size
        self.stats["payload_bytes"] += len(payload)
        self.stats["sent_frames"] += 1
        self.queue_free = max(self.queue_free, release) + size/self.rate
        # Advance channel state even across intentional omissions. Random outcomes
        # depend on application slots, not number of transmitted packets.
        slot = window_id*5+position
        if slot <= self.last_slot:
            raise ValueError("Deliver frames in increasing application-slot order")
        for _ in range(self.last_slot+1, slot+1):
            if self.burst:
                if self.bad:
                    self.bad = self.rng.random() >= 1/3
                else:
                    self.bad = self.rng.random() < min(1., self.loss/(3*max(1-self.loss, 1e-8)))
                lost = self.bad
            else:
                lost = self.rng.random() < self.loss
            jitter = self.rng.normal(0, self.jitter)
        self.last_slot = slot
        arrival = self.queue_free + max(0., self.delay+jitter)
        if lost:
            self.stats["network_lost"] += 1
            return None
        if arrival > release+self.deadline:
            self.stats["late"] += 1
            return None
        try:
            restored = self.aead.decrypt(wire[5:17], wire[17:], wire[:5])
        except Exception:
            self.stats["auth_failed"] += 1
            return None
        assert struct.unpack("!IB", wire[:5]) == (window_id, position)
        return unpack_coeff(restored, len(coeff))
