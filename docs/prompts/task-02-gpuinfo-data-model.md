# Task 2: Update GpuInfo Data Model and `get_gpu_info()` RPC

## Objective

Extend the `GpuInfo` dataclass in `gpu.py` and the `get_gpu_info()` RPC method in `app_controller.py` to expose GPU vendor information (NVIDIA vs AMD) to the frontend.

## Context

Task 1 added AMD GPU detection functions to `gpu.py` (`detect_gpu_vendor()`, `_check_rocm_libs_available()`, `get_rocm_status()`). This task wires those into the data model and RPC layer so the frontend can display vendor-specific GPU status.

The `get_gpu_info()` method in `app_controller.py` is called by the frontend via RPC and returns a dict. The `GpuInfo` dataclass in `gpu.py` defines the shape of that data.

## Files to Modify

- `src-pyloid/services/gpu.py` — update `GpuInfo` dataclass, add `get_rocm_status()` if not already present
- `src-pyloid/app_controller.py` — update `get_gpu_info()` method and its imports

## Requirements

### 2a. Update `GpuInfo` dataclass in `gpu.py`

Add three new fields to the existing dataclass at `src-pyloid/services/gpu.py:60`:

```python
@dataclass
class GpuInfo:
    cuda_available: bool
    device_count: int
    gpu_name: Optional[str]
    supported_compute_types: list[str]
    current_device: str
    current_compute_type: str
    cudnn_available: bool        # NEW
    cudnn_message: Optional[str] # NEW
    gpu_vendor: Optional[str]    # NEW: "nvidia", "amd", or None
    rocm_available: bool         # NEW
    rocm_message: Optional[str]  # NEW
```

Note: The existing dataclass is missing `cudnn_available` and `cudnn_message` — those were already being returned by `get_gpu_info()` in the dict but weren't in the dataclass. Add them too for consistency.

### 2b. Add `get_rocm_status()` function in `gpu.py`

If Task 1 didn't already add this, create:

```python
def get_rocm_status() -> tuple[bool, Optional[str]]:
    """Get ROCm installation status for display in UI."""
    available, error = _check_rocm_libs_available()
    if available:
        if _check_rocm_ctranslate2():
            return True, "ROCm available"
        return False, "ROCm libraries found but ctranslate2 not built with ROCm support"
    return False, error or "ROCm not found"
```

### 2c. Update `get_gpu_info()` in `app_controller.py`

The method is at `src-pyloid/app_controller.py:321`. Update it to:

1. Import `detect_gpu_vendor` and `get_rocm_status` from `services.gpu` (add to the existing import on line 18)
2. Call `detect_gpu_vendor()` to get the vendor
3. Call `get_rocm_status()` to get ROCm status
4. Add three new keys to the returned dict:
   - `"gpuVendor"`: the vendor string (`"nvidia"`, `"amd"`, or `None`)
   - `"rocmAvailable"`: boolean
   - `"rocmMessage"`: string or `None`

The updated return dict should look like:

```python
return {
    "cudaAvailable": cuda_available,
    "deviceCount": 1 if cuda_available else 0,
    "gpuName": gpu_name,
    "supportedComputeTypes": get_cuda_compute_types() if cuda_available else [],
    "currentDevice": self.transcription_service.get_current_device(),
    "currentComputeType": self.transcription_service.get_current_compute_type(),
    "cudnnAvailable": cudnn_available,
    "cudnnMessage": cudnn_message,
    "gpuVendor": vendor,
    "rocmAvailable": rocm_available,
    "rocmMessage": rocm_message,
}
```

## What NOT to Change

- Do NOT modify any frontend files (that's Tasks 3-5)
- Do NOT change the transcription service
- Do NOT rename or remove any existing dict keys in the return value
- Do NOT modify any other RPC methods

## Testing Requirements

Add tests to `src-pyloid/tests/test_gpu_amd.py` (or a new file `test_gpu_info.py`):

1. **`test_get_gpu_info_includes_vendor_fields`** — Mock GPU functions and call `get_gpu_info()` on a controller instance. Verify the returned dict contains `gpuVendor`, `rocmAvailable`, and `rocmMessage` keys.

2. **`test_get_gpu_info_nvidia_scenario`** — Mock as NVIDIA system. Verify `gpuVendor` is `"nvidia"`, `rocmAvailable` is `False`.

3. **`test_get_gpu_info_amd_scenario`** — Mock as AMD system with ROCm. Verify `gpuVendor` is `"amd"`, `rocmAvailable` is `True`.

4. **`test_get_gpu_info_no_gpu`** — Mock no GPU. Verify `gpuVendor` is `None`, both `rocmAvailable` and `cudnnAvailable` are `False`.

5. **`test_get_rocm_status_fully_available`** — Returns `(True, "ROCm available")`
6. **`test_get_rocm_status_libs_no_ct2`** — Returns `(False, "...ctranslate2 not built...")`
7. **`test_get_rocm_status_no_rocm`** — Returns `(False, error_message)`

## Verification Steps

Before considering this task complete, verify:

1. **Tests pass**: Run `cd VoiceFlow && uv run -p .venv pytest src-pyloid/tests/test_gpu_amd.py -v` — all tests must pass
2. **Existing tests still pass**: Run `cd VoiceFlow && uv run -p .venv pytest src-pyloid/tests/ -v --ignore=src-pyloid/tests/test_transcription.py` — no regressions
3. **Import check**: Run `cd VoiceFlow && uv run -p .venv python -c "from services.gpu import GpuInfo, get_rocm_status, detect_gpu_vendor; print('OK')"` — must print OK
4. **No syntax errors in both files**:
   - `uv run -p .venv python -m py_compile src-pyloid/services/gpu.py`
   - `uv run -p .venv python -m py_compile src-pyloid/app_controller.py`
5. **Backward compatibility**: The dict returned by `get_gpu_info()` must still contain all original keys (`cudaAvailable`, `deviceCount`, `gpuName`, `supportedComputeTypes`, `currentDevice`, `currentComputeType`, `cudnnAvailable`, `cudnnMessage`) — the new keys are additions, not replacements
