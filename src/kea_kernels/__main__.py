"""Transcribe 16 kHz audio files with Kea-B or Kea-T on their GPU kernels.

    python -m kea_kernels MODEL_DIR clip.wav [clip2.wav ...]
"""
from __future__ import annotations

import argparse
import json
import signal
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kea-kernels", description=__doc__.split("\n\n")[0])
    p.add_argument("model", help="a downloaded model directory (hf download audeolabs/kea-b-0.1 --local-dir DIR)")
    p.add_argument("audio", nargs="+", help="16 kHz audio files (wav, flac, ...)")
    p.add_argument("--mode", default=None,
                   help="Kea-B: x1, x2, x3; Kea-T: x2, x3, compact (default: the model's own; see USAGE.md)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokens", action="store_true", help="JSON lines with the text, token IDs and frame timestamps")
    p.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check of the model files")
    args = p.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # unwind: finish GPU work and release graphs
    try:
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(nemo_logging.ERROR)
    except ImportError:
        pass
    import torch
    import kea_kernels
    code = 0
    try:
        model = kea_kernels.load(args.model, device=args.device, mode=args.mode, verify=not args.no_verify)
        try:
            for path in args.audio:
                result = kea_kernels.transcribe(model, path, return_tokens=True)
                if args.tokens:
                    print(json.dumps({"audio": path, "text": result.text, "tokens": result.tokens,
                                      "timestamps": result.timestamps, "model": model.name, "mode": model.mode}),
                          flush=True)
                else:
                    print(f"{path}\t{result.text}" if len(args.audio) > 1 else result.text, flush=True)
        finally:
            kea_kernels.close(model)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 1
    finally:
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
    return code


if __name__ == "__main__":
    sys.exit(main())
