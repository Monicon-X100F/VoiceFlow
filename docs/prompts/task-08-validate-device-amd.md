# Task 8: Update `validate_device_setting()` for AMD

## Objective

Update `validate_device_setting()` in `gpu.py` to provide AMD-specific error messages when a user selects "cuda" device but ROCm is not properly configured.

## Context

The `validate_device_setting()` function (at `src-pyloid/services/gpu.py:318`) validates device settings before they're saved. Currently it only gives NVIDIA-specific error messages. With AMD GPU support (Task 1), it needs to give relevant guidance based on the detected GPU vendor.

The function is called from the frontend when a user changes the device setting, via `app_controller.py:validate_device()`.

## File to Modify

- `src-pyloid/services/gpu.py` — update `validate_device_setting()` function

## Requirements

Update the `validate_device_setting()` function so that when `device == "cuda"` and `is_cuda_available()` returns `False`, it provides vendor-specific error messages:

```python
def validate_device_setting(device: str) -> tuple[bool, Optional[str]]:
    if device not in DEVICE_OPTIONS:
        return False, f"Invalid device option: {device}"

    if device == "cuda":
        if not is_cuda_available():
            vendor = detect_gpu_vendor()
            if vendor == "amd":
                rocm_ok, rocm_msg = _check_rocm_libs_available()
                if not rocm_ok:
                    return False, "AMD GPU detected but ROCm is not installed. Install ROCm toolkit or use CPU mode."
                if not _check_rocm_ctranslate2():
                    return False, "ROCm installed but ctranslate2 lacks ROCm support. Run scripts/setup-rocm.sh."
                return False, "AMD GPU and ROCm detected but CUDA acceleration is not working."
            elif vendor == "nvidia":
                # Keep existing NVIDIA error logic
                try:
                    import ctranslate2
                    compute_types = ctranslate2.get_supported_compute_types("cuda")
                    if len(compute_types) > 0:
                        return False, "CUDA detected but cuDNN is not installed. Install cuDNN 9.x or use CPU mode."
                except Exception:
                    pass
                return False, "CUDA is not available. Please install NVIDIA drivers and CUDA toolkit."
            else:
                return False, "No compatible GPU detected. Use CPU mode."

    return True, None
```

Key behaviors:
- **AMD, no ROCm libs**: "AMD GPU detected but ROCm is not installed. Install ROCm toolkit or use CPU mode."
- **AMD, ROCm libs present, wrong ctranslate2**: "ROCm installed but ctranslate2 lacks ROCm support. Run scripts/setup-rocm.sh."
- **AMD, everything looks OK but still failing**: "AMD GPU and ROCm detected but CUDA acceleration is not working."
- **NVIDIA scenarios**: preserve existing behavior exactly
- **No GPU**: "No compatible GPU detected. Use CPU mode."

## What NOT to Change

- Do NOT modify the function signature
- Do NOT change behavior for `device == "auto"` or `device == "cpu"` — those should still return `(True, None)`
- Do NOT modify any other functions in `gpu.py`
- Do NOT modify any other files

## Testing Requirements

Add tests to `src-pyloid/tests/test_gpu_amd.py`:

1. **`test_validate_cuda_with_working_amd`** — Mock AMD fully configured, `is_cuda_available()` returns `True` → `(True, None)`
2. **`test_validate_cuda_amd_no_rocm`** — Mock AMD GPU, no ROCm libs → `(False, "...ROCm is not installed...")`
3. **`test_validate_cuda_amd_no_ct2`** — Mock AMD GPU, ROCm libs OK, standard ct2 → `(False, "...ctranslate2 lacks ROCm...")`
4. **`test_validate_cuda_nvidia_no_cudnn`** — Mock NVIDIA GPU, no cuDNN → existing error message preserved
5. **`test_validate_cuda_no_gpu`** — No GPU detected → `(False, "No compatible GPU detected...")`
6. **`test_validate_cpu_always_valid`** — `device="cpu"` → `(True, None)` regardless of GPU state
7. **`test_validate_auto_always_valid`** — `device="auto"` → `(True, None)` regardless of GPU state

## Verification Steps

Before considering this task complete, verify:

1. **Tests pass**: Run `cd VoiceFlow && uv run -p .venv pytest src-pyloid/tests/test_gpu_amd.py -v -k "validate"` — all validation tests pass
2. **All tests pass**: Run `cd VoiceFlow && uv run -p .venv pytest src-pyloid/tests/test_gpu_amd.py -v` — no regressions
3. **No syntax errors**: Run `cd VoiceFlow && uv run -p .venv python -m py_compile src-pyloid/services/gpu.py`
4. **Error messages are helpful**: Read each error message and verify it tells the user what to do (install X, run Y, or use CPU)
5. **Existing NVIDIA path preserved**: Verify the NVIDIA branch still checks ctranslate2 compute types to distinguish "CUDA but no cuDNN" from "no CUDA at all"
