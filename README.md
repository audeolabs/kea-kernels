# kea-kernels

The GPU kernels behind Kea-B and Kea-T, Audeo Labs' small English speech recognizers. The models are
NVIDIA's Parakeet-TDT-0.6B-v2 with almost every weight stored in one bit (Kea-B) or two (Kea-T). The
weights stay that small on the GPU, and these kernels multiply them on INT8 tensor cores, expanding the bits
in registers.

- [Kea-B 0.1](https://huggingface.co/audeolabs/kea-b-0.1): 1-bit weights, 91.2 MB. It averages 6.77% word
  error over seven English test sets, on these kernels. The original model is 2,472 MB and scores 6.44%.
- [Kea-B](https://huggingface.co/audeolabs/kea-b) (0.0): the first 1-bit release, 105.3 MB, 7.05%.
- [Kea-T](https://huggingface.co/audeolabs/kea-t): 2-bit (ternary) weights, 180.8 MB, 6.84%.

On an RTX 5090, Kea-B transcribes a 10-second clip in 3.9 ms using 1.2 GB of GPU memory (from the
[Kea-B card](https://huggingface.co/audeolabs/kea-b)).

```sh
hf download audeolabs/kea-b-0.1 --local-dir kea-b-0.1
python -m kea_kernels kea-b-0.1 clip.wav
```

It runs on the RTX 5090 (SM120) only, for now. See [USAGE.md](USAGE.md) for installation, the Python API
and the options.

MIT licensed. The model weights are CC-BY-4.0, from their own repositories.
