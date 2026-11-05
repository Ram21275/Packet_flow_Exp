import io
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .codec import SR, FRAME, encode_audio


def prepare(root, train_limit=1000, valid_limit=100, test_limit=100, dim=160, local_root=None):
    """HF streaming avoids downloading the entire corpus before the pilot.

    Decode=False + soundfile avoids torchcodec/FFmpeg compatibility issues.
    With local_root, recursively reads extracted LibriSpeech split directories.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError("Cache exists; use a fresh --data directory to change limits or codec")
    manifest = {"sr": SR, "frame": FRAME, "dim": dim, "codec": "dct-int8-v1", "splits": {}}
    sums, squares, count = np.zeros(dim), np.zeros(dim), 0
    for split, hf_split, folder, limit in [
        ("train", "train.100", "train-clean-100", train_limit),
        ("valid", "validation", "dev-clean", valid_limit),
        ("test", "test", "test-clean", test_limit),
    ]:
        if local_root:
            files = sorted((Path(local_root) / folder).rglob("*.flac"))
            if not files:
                raise FileNotFoundError(f"No FLAC files below {local_root}/{folder}")
            source = ({"id": p.stem, "audio_path": p} for p in files)
        else:
            from datasets import Audio, load_dataset
            source = load_dataset("openslr/librispeech_asr", "clean", split=hf_split, streaming=True)
            source = source.cast_column("audio", Audio(decode=False))
        dest = root / split
        dest.mkdir(exist_ok=True)
        entries = []
        for row in tqdm(source, desc=f"cache {split}", total=limit or None):
            if "audio_path" in row:
                audio, sr = sf.read(row["audio_path"], dtype="float32")
            else:
                item = row["audio"]
                audio, sr = sf.read(io.BytesIO(item["bytes"]) if item.get("bytes") else item["path"], dtype="float32")
            if sr != SR or audio.ndim != 1:
                raise ValueError("Expected 16 kHz mono input")
            if len(audio) < FRAME * 35:
                continue
            z = encode_audio(audio, dim)
            name = str(row["id"])
            if Path(name).name != name:
                raise ValueError("Unsafe utterance identifier")
            path = dest / f"{name}.npz"
            np.savez_compressed(path, z=z.astype(np.float32))
            sf.write(dest / f"{name}.wav", audio[:len(z)*FRAME], SR)
            entries.append({"path": str(path.relative_to(root)), "frames": len(z), "id": name})
            if split == "train":
                sums += z.sum(axis=0, dtype=np.float64)
                squares += np.square(z.astype(np.float64)).sum(axis=0)
                count += len(z)
            if limit and len(entries) >= limit:
                break
        if not entries:
            raise ValueError(f"Empty {split} split")
        manifest["splits"][split] = entries
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-6))
    manifest.update(mean=mean.tolist(), std=std.tolist(), train_frames=count)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_manifest(root):
    return json.loads((Path(root) / "manifest.json").read_text())


class Windows(Dataset):
    def __init__(self, root, split, history=25, window=5):
        self.manifest = load_manifest(root)
        self.history, self.window = history, window
        self.mean = np.asarray(self.manifest["mean"], dtype=np.float32)
        self.std = np.asarray(self.manifest["std"], dtype=np.float32)
        self.arrays, self.index = [], []
        for entry in self.manifest["splits"][split]:
            with np.load(Path(root) / entry["path"]) as archive:
                z = (archive["z"] - self.mean) / self.std
            self.arrays.append(z.astype(np.float32))
            i = len(self.arrays) - 1
            for start in range(history, len(z)-window+1, window):
                self.index.append((i, start))
        if not self.index:
            raise ValueError("No windows available")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, start = self.index[idx]
        z = self.arrays[i]
        return torch.from_numpy(z[start-self.history:start].copy()), torch.from_numpy(z[start:start+self.window].copy())


def synthetic(root, dim=16):
    """Distinct synthetic utterances for executable tests, not quality evidence."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(72)
    manifest = {"sr": SR, "frame": FRAME, "dim": dim, "codec": "dct-int8-v1", "mean": [0.] * dim, "std": [1.] * dim, "splits": {}}
    for split in ("train", "valid", "test"):
        dest = root / split
        dest.mkdir(exist_ok=True)
        entries = []
        for i in range(3):
            t = np.arange(320*80) / SR
            x = (.12*np.sin(2*np.pi*rng.uniform(150, 350)*t) + .02*rng.normal(size=len(t))).astype(np.float32)
            z = encode_audio(x, dim)
            path = dest / f"{i}.npz"
            np.savez_compressed(path, z=z)
            sf.write(dest / f"{i}.wav", x, SR)
            entries.append({"path": str(path.relative_to(root)), "frames": len(z), "id": str(i)})
        manifest["splits"][split] = entries
    (root / "manifest.json").write_text(json.dumps(manifest))
