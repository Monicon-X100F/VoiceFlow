# Task 7: Update Linux Library Preloading in `main.py`

## Objective

Extend the Linux GPU library preloading in `main.py` to also preload ROCm libraries if present, alongside the existing NVIDIA library preloading.

## Context

At the top of `src-pyloid/main.py` (lines 7-29), there's a Linux-specific block that preloads NVIDIA `.so` libraries from pip packages before any CUDA imports. This ensures ctranslate2/faster-whisper can find them at runtime. The same approach is needed for ROCm libraries, which are installed system-wide under `/opt/rocm/`.

This code runs at module import time, before the Pyloid app is created.

## File to Modify

- `src-pyloid/main.py` — the Linux-specific preload section at the top of the file (inside the `if sys.platform.startswith('linux'):` block)

## Requirements

### 7a. Rename and refactor the preload function

Rename `_preload_nvidia_libs()` to `_preload_gpu_libs()` and have it call two sub-functions:

```python
def _preload_gpu_libs():
    """Preload GPU .so libs so ctranslate2 can find them."""
    _preload_nvidia_pip_libs()
    _preload_rocm_libs()
```

Move the existing `_preload_nvidia_libs()` body into `_preload_nvidia_pip_libs()`.

Update the call site (currently line 27: `_preload_nvidia_libs()`) to call `_preload_gpu_libs()`.

### 7b. Add `_preload_rocm_libs()` function

```python
def _preload_rocm_libs():
    """Preload ROCm .so libraries if available."""
    import ctypes
    rocm_lib_paths = [
        "/opt/rocm/lib",
        "/opt/rocm/hip/lib",
    ]
    rocm_libs = ["libamdhip64.so", "librocblas.so", "libhipblas.so"]

    for lib_path in rocm_lib_paths:
        if not os.path.isdir(lib_path):
            continue
        for lib_name in rocm_libs:
            full_path = os.path.join(lib_path, lib_name)
            if os.path.exists(full_path):
                try:
                    ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
```

### 7c. Add ROCm environment variables

Before the `_preload_gpu_libs()` call, add ROCm environment setup:

```python
# Help ROCm find its libraries
if os.path.isdir("/opt/rocm"):
    rocm_path = "/opt/rocm"
    os.environ.setdefault("ROCM_PATH", rocm_path)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    rocm_lib = os.path.join(rocm_path, "lib")
    if rocm_lib not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = rocm_lib + ":" + ld_path
```

### Important: Error handling

All new code must be wrapped in try/except like the existing preload call (line 26-29). The preload is best-effort — it must never crash the app on startup.

## What NOT to Change

- Do NOT modify any code outside the `if sys.platform.startswith('linux'):` block at the top
- Do NOT change the Hyprland rules, Qt WebEngine flags, or any other Linux-specific setup
- Do NOT modify any other files
- Do NOT change the order of operations — NVIDIA preloading should still happen first

## Verification Steps

Before considering this task complete, verify:

1. **No syntax errors**: Run `cd VoiceFlow && uv run -p .venv python -m py_compile src-pyloid/main.py` — must succeed (note: this may fail due to import issues in the test environment, so also check manually)
2. **Read the modified file**: Verify the structure is:
   - `_preload_nvidia_pip_libs()` contains the original NVIDIA logic
   - `_preload_rocm_libs()` contains the new ROCm logic
   - `_preload_gpu_libs()` calls both
   - ROCm env vars are set before `_preload_gpu_libs()` call
   - Everything is wrapped in try/except
3. **NVIDIA logic preserved**: Compare the body of `_preload_nvidia_pip_libs()` to the original `_preload_nvidia_libs()` — they should be identical
4. **No global side effects**: Verify `_preload_rocm_libs()` gracefully handles the case where `/opt/rocm` doesn't exist (it should just return without doing anything)
5. **Function call updated**: The old `_preload_nvidia_libs()` call is replaced with `_preload_gpu_libs()`
