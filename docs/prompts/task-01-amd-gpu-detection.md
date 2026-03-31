# Task 1: Add AMD GPU Detection to `gpu.py`

## Objective

Extend `src-pyloid/services/gpu.py` to detect AMD GPUs and check ROCm library availability, alongside the existing NVIDIA-only detection. This is the foundation for AMD GPU support — all other tasks depend on this.

## Context

VoiceFlow uses faster-whisper (backed by CTranslate2) for transcription. Currently, GPU acceleration only supports NVIDIA via CUDA/cuDNN. CTranslate2 has a community ROCm/HIP fork that uses `device="cuda"` through HIP's CUDA compatibility layer, so the transcription code itself needs no changes — only the detection, library checking, and status reporting need to become vendor-aware.

**Key technical detail:** When built with ROCm/HIP, CTranslate2 still uses `device="cuda"` — HIP provides a CUDA compatibility layer. This means `DEVICE_OPTIONS` stays `["auto", "cpu", "cuda"]`.

## Files to Modify

- `src-pyloid/services/gpu.py` — main changes
- `src-pyloid/tests/test_gpu_amd.py` — new test file

## Requirements

### 1a. Add `detect_gpu_vendor()` function

Create a new public function that returns `"nvidia"`, `"amd"`, or `None`:

```python
def detect_gpu_vendor() -> Optional[str]:
```

- Check NVIDIA first (try running `nvidia-smi` via subprocess, timeout 5s)
- If NVIDIA not found, check AMD:
  - Try `rocm-smi --showproductname` (subprocess, timeout 5s)
  - If `rocm-smi` not found, fall back to `lspci | grep -i "VGA.*AMD\|Display.*AMD"` (Linux only)
- Cache the result similarly to `_cuda_available_cache` to avoid repeated subprocess calls
- Use the existing `log` logger with domain "gpu"

### 1b. Add `_check_rocm_libs_available()` function

Similar pattern to `_check_cudnn_available()`:

```python
def _check_rocm_libs_available() -> tuple[bool, Optional[str]]:
```

- Use `ctypes.CDLL` to check for: `libamdhip64.so`, `librocblas.so`, `libhipblas.so`
- Return `(True, None)` if all load successfully
- Return `(False, "ROCm library {name} not found. Install ROCm toolkit.")` on first failure

### 1c. Add `_check_rocm_ctranslate2()` function

Verify the installed ctranslate2 was built with ROCm/HIP support:

```python
def _check_rocm_ctranslate2() -> bool:
```

- Import ctranslate2 and call `get_supported_compute_types("cuda")`
- ROCm-built ctranslate2 reports "cuda" compute types via HIP compatibility
- Return `True` if compute types are non-empty, `False` otherwise
- Wrap in try/except, return `False` on any exception

### 1d. Extend `get_gpu_name()` for AMD

After the existing nvidia-smi attempt fails, try:
- Run `rocm-smi --showproductname` (subprocess, timeout 5s)
- Parse the output for the GPU product name
- Return the name string or fall through to `None`

### 1e. Update `is_cuda_available()` to be vendor-aware

Modify the existing function so it works for both vendors:
- Keep the existing ctranslate2 CUDA compute types check at the top
- After confirming ctranslate2 sees a CUDA device, branch on vendor:
  - `"nvidia"` → check cuDNN (existing `_check_cudnn_available()`)
  - `"amd"` → check ROCm libs AND ROCm ctranslate2
  - If vendor is `None` but ctranslate2 sees CUDA, fall through to the existing cuDNN check (backward compat)
- Maintain the `_cuda_available_cache` behavior

### 1f. Add `ROCM_COMPUTE_TYPE` constant

```python
ROCM_COMPUTE_TYPE = "float16"
```

No changes to `DEVICE_OPTIONS` — device strings remain `"auto"/"cpu"/"cuda"` since HIP maps to the CUDA device string.

## What NOT to Change

- Do NOT modify `TranscriptionService` or any transcription code
- Do NOT change `DEVICE_OPTIONS` values
- Do NOT add new pip dependencies (ROCm comes from system packages)
- Do NOT modify any frontend files

## Testing Requirements

Create `src-pyloid/tests/test_gpu_amd.py` with these test cases (all using mocks — no real GPU required):

1. **`TestDetectGpuVendor`**:
   - `test_nvidia_gpu_detected` — mock nvidia-smi succeeding → returns `"nvidia"`
   - `test_amd_gpu_detected_via_rocm_smi` — mock nvidia-smi failing, rocm-smi succeeding → returns `"amd"`
   - `test_amd_gpu_detected_via_lspci` — mock both smi tools failing, lspci showing AMD → returns `"amd"`
   - `test_no_gpu` — all detection methods fail → returns `None`
   - `test_nvidia_preferred_over_amd` — both present → returns `"nvidia"` (checked first)

2. **`TestRocmLibsAvailable`**:
   - `test_all_libs_present` — mock ctypes.CDLL succeeding for all 3 libs → `(True, None)`
   - `test_missing_lib` — mock one CDLL raising OSError → `(False, error_message)`

3. **`TestRocmCtranslate2`**:
   - `test_rocm_ct2_available` — mock ctranslate2 returning compute types → `True`
   - `test_standard_ct2` — mock ctranslate2 raising exception → `False`

4. **`TestIsCudaAvailableAmd`**:
   - `test_amd_with_rocm_and_ct2` — full AMD setup → `True`
   - `test_amd_without_rocm` — AMD GPU, no ROCm libs → `False`
   - `test_amd_with_rocm_no_ct2` — ROCm libs OK, standard ctranslate2 → `False`

5. **`TestGetGpuNameAmd`**:
   - `test_amd_gpu_name_from_rocm_smi` — mock rocm-smi output → correct name
   - `test_fallback_to_none` — both nvidia-smi and rocm-smi fail → `None`

Mock `subprocess.run` for all smi/lspci calls. Mock `ctypes.CDLL` for library detection. Reset `_cuda_available_cache` and any vendor cache in each test's setup.

## Verification Steps

Before considering this task complete, verify:

1. **Tests pass**: Run `cd VoiceFlow && uv run -p .venv pytest src-pyloid/tests/test_gpu_amd.py -v` — all tests must pass
2. **Existing tests still pass**: Run `cd VoiceFlow && uv run -p .venv pytest src-pyloid/tests/ -v --ignore=src-pyloid/tests/test_transcription.py` — no regressions
3. **Import check**: Run `cd VoiceFlow && uv run -p .venv python -c "from services.gpu import detect_gpu_vendor, get_rocm_status; print('OK')"` — must print OK
4. **No syntax errors**: Run `cd VoiceFlow && uv run -p .venv python -m py_compile src-pyloid/services/gpu.py` — must succeed with no output
5. **Existing public API preserved**: Verify that `is_cuda_available`, `get_gpu_name`, `get_cuda_compute_types`, `validate_device_setting`, `get_cudnn_status`, `reset_cuda_cache`, `has_nvidia_gpu` all still exist and are importable
