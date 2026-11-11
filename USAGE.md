# Using kea-kernels

## Requirements

- **GPU:** an NVIDIA RTX 5090 (compute capability 12.0, "SM120"). The kernels use SM120 instructions and
  tile tables tuned on that card, and no other GPU has been tested. `load()` raises a `ValueError` on any
  other device.
- **Driver:** one that supports CUDA 12.8 (R570 or newer).
- **Linux, Python 3.11 or newer, and a C compiler** (`gcc` or `clang`, for example `apt install gcc`). Triton
  uses it once, to build a small launcher module that it then caches.
- **Packages** (installed with the package; the versions we measured are in brackets): PyTorch 2.11 built for
  CUDA 12.8 [2.11.0+cu128], Triton 3.6 [3.6.0], NeMo 3.0 with its ASR extras [3.0.0], cuda-bindings 12.9,
  NVRTC 12 from `nvidia-cuda-nvrtc-cu12` [12.8.93], safetensors, soundfile and omegaconf.

No CUDA toolkit is needed. The 1-bit kernels are compiled at run time by NVRTC (from the pip wheel), and the
rest by Triton.

## Install

Into a fresh virtual environment, taking PyTorch from its CUDA 12.8 index:

```sh
python -m venv .venv && . .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    "kea-kernels @ git+https://github.com/audeolabs/kea-kernels"
```

Or with uv:

```sh
uv venv && . .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match "kea-kernels @ git+https://github.com/audeolabs/kea-kernels"
```

NeMo pulls in many dependencies, so the install takes a few minutes and about 7 GB.

## Download a model

```sh
pip install huggingface_hub   # for the `hf` command, if you don't have it
hf download audeolabs/kea-b-0.1 --local-dir kea-b-0.1
hf download audeolabs/kea-b     --local-dir kea-b
hf download audeolabs/kea-t     --local-dir kea-t
```

Pass the downloaded directory to kea-kernels. It works out which model is in it. For Kea-B 0.1 it reads
the `compat/` export inside the download (the same weights as `bundle.safetensors`, in the layout the kernels
read).

## Transcribe from the command line

```sh
python -m kea_kernels kea-b-0.1 clip.wav                 # prints the transcript
python -m kea_kernels kea-b-0.1 a.wav b.flac c.wav       # one "path<TAB>text" line per file
python -m kea_kernels kea-t --mode x3 clip.wav           # another mode (see below)
python -m kea_kernels kea-b --tokens clip.wav            # JSON lines: text, token IDs, frame timestamps
```

The `kea-kernels` command is the same program. Audio must be 16 kHz: any format soundfile reads (WAV, FLAC,
OGG...), with several channels averaged to mono. To convert, run `ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`.
Errors go to stderr and exit with status 1.

## Transcribe from Python

```python
import kea_kernels

model = kea_kernels.load("kea-b-0.1")                    # device="cuda", mode=None (the model's default)
print(kea_kernels.transcribe(model, "clip.wav"))         # a str
texts = kea_kernels.transcribe(model, ["a.wav", "b.wav"])   # a list, in input order
result = kea_kernels.transcribe(model, samples, return_tokens=True)
# samples: a 1-D float array or tensor of 16 kHz mono audio in [-1, 1]
# result.text, result.tokens (SentencePiece IDs), result.timestamps (encoder frames, 80 ms each)
kea_kernels.close(model)
```

`load()` returns a `Model`, which also works as a context manager (`with kea_kernels.load(...) as model:`)
and has `.transcribe(...)`. `model.nemo` is the underlying NeMo `EncDecRNNTBPEModel`, transformed for
inference. Don't train it, move it or save it. `close()` releases its CUDA graphs and undoes the
transforms.

Transcription runs one clip at a time (batch 1), the way the kernels were tuned and measured. Each clip
is decoded in one pass with full attention. The models were trained on clips of 1 to 30 s, so split longer
recordings (ideally at pauses) into pieces of about 30 s or less. Memory grows with clip length. Clips
shorter than 30 ms are zero-padded to 30 ms.

## Modes

A mode sets how many INT8 parts each activation is split into before it meets the low-bit weights (x1, x2
or x3), and, for Kea-T, how the weights are kept on the GPU. The default for each model is the mode its
published numbers were measured in:

| Model | Mode | What runs | |
| --- | --- | --- | --- |
| Kea-B 0.1 | **x1** | one INT8 part per frame ("i8x1"), the mode Kea-B 0.1 was trained for | default |
| Kea-B 0.0 | **x2** | two residual INT8 parts, fused row epilogues | default |
| Kea-B (either) | x3 | three residual INT8 parts, the earlier Kea-B 0.0 runtime | |
| Kea-T | **x2** | two parts, with an INT8 copy of the 2-bit codes (about 604 MB more GPU memory) | default |
| Kea-T | x3 | three parts, same storage as x2 | |
| Kea-T | compact | the codes stay 2-bit on the GPU (three parts): the least memory, a little slower | |

Kea-B 0.1 also accepts x2 and x3, and Kea-B 0.0 accepts x1. Those combinations run, but the cards
publish no numbers for them.

## What to expect

kea-kernels reproduces the research runtime the model cards were measured with. On 40 LibriSpeech
dev-other clips, every mode above gave the same tokens and timestamps as that runtime. So accuracy is
what the cards report for kernel mode:

- **Kea-B 0.1, x1:** a 6.77% mean test WER over the seven sets, the same as its FP32 numbers to within
  0.05 points per set ([card](https://huggingface.co/audeolabs/kea-b-0.1)).
- **Kea-B 0.0, x2 (or x3):** a 6.3122% (6.3120%) mean dev WER, the same as its FP32 decoding. Its test sets
  were scored in FP32 only (7.05%) ([card](https://huggingface.co/audeolabs/kea-b)).
- **Kea-T, x2 (or x3):** a 6.0853% (6.0856%) mean dev WER, with 6.0856% in FP32. Its test sets were
  scored in FP32 only (6.84%) ([card](https://huggingface.co/audeolabs/kea-t)).

Speed and memory are in each card's runtime section. All figures are from one RTX 5090 with a 400 W
power limit.

The first transcription after `load()` is slow: the kernels for each input length are compiled (NVRTC
and Triton, cached on disk) and CUDA graphs are captured (cached in memory, for up to four lengths). Later
calls with a cached length are fast. Each model loads in a few seconds plus NeMo's import time.

## Caches and environment variables

| Variable | Effect |
| --- | --- |
| `KEA_KERNELS_CACHE` | where compiled 1-bit kernels are cached (default `$XDG_CACHE_HOME/kea-kernels`, else `~/.cache/kea-kernels`) |
| `TRITON_CACHE_DIR` | Triton's kernel cache (default `~/.triton/cache`; about 430 MB after all six modes) |
| `CUDA_HOME` | only read by PyTorch's NVRTC helper; if it's unset, kea-kernels points it at an empty directory in its cache |
| `CUDA_VISIBLE_DEVICES` | which GPU; or pass `device="cuda:1"` |

The caches can be deleted at any time. If the cache directory can't be written, kernels are recompiled in
each process. kea-kernels downloads nothing: it reads only the directory you give it, and it checks every
file against the manifest's SHA-256 (`verify=False` or `--no-verify` skips this).

## Troubleshooting

- **`this runtime targets NVIDIA SM120 (RTX 5090)`**: your GPU isn't an RTX 5090. To use the models on
  other hardware, run the standalone loader in each model download (`load_kea_b.py`, `load_binary.py`,
  `load_ternary.py`). It rebuilds an ordinary FP32 NeMo model and works on any GPU or CPU, with the same
  accuracy, but not the memory or speed savings.
- **`Failed to find C compiler`**: install `gcc` (or set `CC` to a C compiler). Triton needs one the first
  time it runs.
- **`NVRTC 12 was not found`**: `pip install nvidia-cuda-nvrtc-cu12`, or put `libnvrtc.so.12` on
  `LD_LIBRARY_PATH`.
- **Slow decoding:** `kea_kernels.decoder_graphs_mode(model)` should return `full_graph` after the first
  transcription. If it doesn't, NeMo couldn't use NVRTC or cuda-bindings and fell back to a slower decoder
  loop. Check that `cuda-bindings` (12.9) is installed and that NVRTC loads.
- **`CUDA error: out of memory` on long audio:** split it into pieces of 30 s or less.
- **`... does not match the manifest SHA-256`**: the download is incomplete or modified. Run
  `hf download` again.
- **`unsupported format`**: pass the model directory itself (the one with `manifest.json`), not a file in it.
- **NixOS and other systems without an FHS layout:** PyTorch's and Triton's pip wheels expect a standard
  `/lib64` loader and `libstdc++`. Run inside an FHS environment (for example `buildFHSEnv`) or with nix-ld.
