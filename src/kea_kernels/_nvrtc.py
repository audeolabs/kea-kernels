"""NVRTC compilation, an on-disk cubin cache and a typed cuLaunchKernel launcher (SM120).

The 1-bit kernels use no CUDA headers. NVRTC is loaded once, globally, from the `nvidia-cuda-nvrtc-cu12` wheel if it
is installed, else from the system library path (`libnvrtc.so.12`). Loading it globally also lets NeMo's full
CUDA-graph decoder find it.

Compiled cubins are cached under `cache_dir()`, keyed by the SHA-256 of (source, kernel name, architecture,
options, NVRTC version). Compilation is refused during CUDA-graph capture. `Kernel.launch` packs every argument
with an explicit C type and enqueues on torch.cuda.current_stream() unless a stream is given.
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import tempfile
from pathlib import Path

import torch

OPTIONS = ("--std=c++17", "--fmad=false")
SONAME = "libnvrtc.so.12"


def cache_dir() -> Path:
    """$KEA_KERNELS_CACHE, else $XDG_CACHE_HOME/kea-kernels, else ~/.cache/kea-kernels."""
    explicit = os.environ.get("KEA_KERNELS_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "kea-kernels"


def _candidates() -> list[str]:
    """NVRTC libraries to try, in order: the pip wheel's, then the dynamic loader's search path."""
    found = []
    try:
        import nvidia.cuda_nvrtc  # the nvidia-cuda-nvrtc-cu12 wheel (a namespace package)
        for root in getattr(nvidia.cuda_nvrtc, "__path__", []):
            path = Path(root) / "lib" / SONAME
            if path.is_file():
                found.append(str(path))
    except ImportError:
        pass
    found.append(SONAME)
    return found


@functools.lru_cache(None)
def setup() -> str:
    """Load NVRTC globally and make it usable by torch.cuda._utils; return its version string."""
    errors = []
    lib = None
    for candidate in _candidates():
        try:
            lib = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    if lib is None:
        raise RuntimeError("NVRTC 12 was not found. Install it with `pip install nvidia-cuda-nvrtc-cu12` or put "
                           f"{SONAME} on the library path. Tried: " + "; ".join(errors))
    if not os.environ.get("CUDA_HOME"):
        # torch.cuda._utils wants a CUDA_HOME; the kernels include no headers, so an empty one is enough.
        home = cache_dir() / "cuda-headerless"
        (home / "include").mkdir(parents=True, exist_ok=True)
        os.environ["CUDA_HOME"] = str(home)
    from torch.utils import cpp_extension
    if cpp_extension.CUDA_HOME is None:
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]
    major, minor = ctypes.c_int(), ctypes.c_int()
    lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    return f"{major.value}.{minor.value}"


def cubin_key(source: str, name: str, arch: str, options, version: str) -> str:
    return hashlib.sha256("\0".join([source, name, arch, *options, version]).encode()).hexdigest()


def compile_cubin(source: str, name: str, arch: str = "120a", options: tuple[str, ...] = OPTIONS) -> bytes:
    version = setup()
    path = cache_dir() / "cubin" / f"{cubin_key(source, name, arch, options, version)}.cubin"
    if path.is_file():
        return path.read_bytes()
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("compile kernels before CUDA graph capture (warm up first)")
    from torch.cuda._utils import _nvrtc_compile
    binary, symbol = _nvrtc_compile(source, name, arch, nvcc_options=list(options))
    if symbol != name:
        raise RuntimeError(f'expected extern "C" symbol {name!r}, got {symbol!r}')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(binary)
        os.replace(tmp, path)
    except OSError:
        pass  # an unwritable cache only costs a recompile next time
    return binary


class Kernel:
    """One loaded kernel; `launch` is CUDA-graph capturable (no sync, no allocation)."""

    def __init__(self, cubin: bytes, name: str, shared_bytes: int):
        from torch.cuda._utils import _cuda_load_module, _get_gpu_runtime_library
        self.cubin = cubin
        self.name = name
        self.kernel = _cuda_load_module(cubin, [name])[name]
        self.lib = _get_gpu_runtime_library()
        self.shared_bytes = shared_bytes
        if shared_bytes >= 48 * 1024:
            self.kernel.set_shared_memory_config(shared_bytes)

    def launch(self, grid, block, args, stream=None):
        """args: sequence of (ctype, value); tensors pass as (ctypes.c_void_p, t.data_ptr())."""
        holders = [ctype(value) for ctype, value in args]
        array = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        stream = torch.cuda.current_stream() if stream is None else stream
        result = self.lib.cuLaunchKernel(self.kernel.func, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                                         self.shared_bytes, stream._as_parameter_, array, None)
        if result != 0:
            raise RuntimeError(f"cuLaunchKernel({self.name}) failed with CUresult {result}")


_LOADED: dict[tuple, Kernel] = {}


def load(source: str, name: str, shared_bytes: int) -> Kernel:
    key = (hashlib.sha256(source.encode()).hexdigest(), name, shared_bytes)
    kernel = _LOADED.get(key)
    if kernel is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("load kernels before CUDA graph capture (warm up first)")
        kernel = _LOADED[key] = Kernel(compile_cubin(source, name), name, shared_bytes)
    return kernel


def ptr(t: torch.Tensor | None) -> tuple:
    return (ctypes.c_void_p, 0 if t is None else t.data_ptr())


def i32(v: int) -> tuple:
    return (ctypes.c_int, int(v))


def i64(v: int) -> tuple:
    return (ctypes.c_longlong, int(v))
