# AMD GPU Support — Task Prompts for Sonnet

Each prompt below is self-contained. Copy one at a time into a new Sonnet conversation. They don't depend on each other except where noted in the "Prerequisites" section (meaning those tasks' code changes must already exist in the codebase before running the prompt).

---

## Task 1: Add AMD GPU Detection (`gpu.py`)

**Prerequisites:** None (this is the foundation task)

**Prompt:**

```
You are working on the VoiceFlow project, a cross-platform voice-to-text app. Your task is to add AMD GPU (ROCm) detection to the existing GPU service.

## Context

VoiceFlow uses faster-whisper (backed by CTranslate2) for transcription. Currently only NVIDIA GPUs are supported. CTranslate2 has no official ROCm wheel, but a community fork builds it with HIP support. When built with ROCm/HIP, CTranslate2 still uses `device="cuda"` (HIP provides a CUDA compatibility layer), so faster-whisper API calls need NO changes.

This means:
- `device="cuda"` works for both NVIDIA and AMD
- GPU detection and library checking need to become vendor-aware
- Users need ROCm system packages + a ROCm-enabled ctranslate2 build

## File to edit

`src-pyloid/services/gpu.py`

Read the file first to understand the existing code structure.

## What to implement

### 1a. Add `detect_gpu_vendor()` function
Returns `"nvidia"`, `"amd"`, or `None`.
- Check NVIDIA first via the existing nvidia-smi approach (extract into a `_check_nvidia_present()` helper)
- Check AMD via `rocm-smi --showproductname` (subprocess, timeout 5s)
- If `rocm-smi` not found, fall back to `lspci | grep -i "VGA.*AMD\|Display.*AMD"` (Linux only)
- Cache the result similarly to `_cuda_available_cache`

### 1b. Add `_check_rocm_libs_available()` function
Returns `tuple[bool, Optional[str]]` — similar pattern to `_check_cudnn_available()`.
Check for key ROCm libraries using ctypes:
- `libamdhip64.so`
- `librocblas.so`
- `libhipblas.so`

If any are missing, return `(False, "ROCm library {name} not found. Install ROCm toolkit.")`.

### 1c. Extend `get_gpu_name()` for AMD
After the existing nvidia-smi attempt fails, try:
```python
result = subprocess.run(
    ["rocm-smi", "--showproductname"],
    capture_output=True, text=True, timeout=5
)
```
Parse the output for the GPU name. Use the same `creationflags` pattern and error handling as the nvidia-smi block.

### 1d. Add `_check_rocm_ctranslate2()` function
Returns `bool`. Verifies ctranslate2 was built with ROCm/HIP support:
```python
def _check_rocm_ctranslate2() -> bool:
    try:
        import ctranslate2
        compute_types = ctranslate2.get_supported_compute_types("cuda")
        return len(compute_types) > 0
    except Exception:
        return False
```

### 1e. Update `is_cuda_available()` to be vendor-aware
The function should:
1. Keep the existing ctranslate2 check at the top
2. After ctranslate2 confirms CUDA types exist, call `detect_gpu_vendor()`
3. For `"nvidia"`: do the existing cuDNN check
4. For `"amd"`: check `_check_rocm_libs_available()` AND `_check_rocm_ctranslate2()`
5. For `None`: return False

Important: keep the cache mechanism working. Clear the vendor cache in `reset_cuda_cache()`.

### 1f. Add `get_rocm_status()` function
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

### 1g. Add constant
```python
ROCM_COMPUTE_TYPE = "float16"
```

No change to `DEVICE_OPTIONS` — device is still "auto"/"cpu"/"cuda" since HIP maps to CUDA device string.

## Code style requirements
- Use the existing logger: `from services.logger import get_logger` and `log = get_logger("gpu")`
- Use the same subprocess patterns (creationflags, timeouts, error handling) as existing code
- Keep type annotations consistent with existing code (`Optional[str]`, `tuple[bool, Optional[str]]`)
- Don't add docstrings to functions that don't already have them in the existing code style — but DO add docstrings to new public functions since the existing code does this

## Acceptance criteria
1. `detect_gpu_vendor()` exists and returns "nvidia", "amd", or None
2. `_check_rocm_libs_available()` checks for the 3 ROCm .so files
3. `get_gpu_name()` tries rocm-smi after nvidia-smi fails
4. `_check_rocm_ctranslate2()` verifies ctranslate2 has CUDA compute types
5. `is_cuda_available()` works for both NVIDIA (cuDNN check) and AMD (ROCm check)
6. `get_rocm_status()` returns status tuple for UI display
7. `reset_cuda_cache()` also resets vendor cache
8. `ROCM_COMPUTE_TYPE = "float16"` constant added
9. All existing NVIDIA functionality is preserved — no regressions

## Verification steps
After making changes:
1. Read through `gpu.py` and verify all functions are syntactically correct
2. Verify that the existing `is_cuda_available()` call path for NVIDIA is preserved (cuDNN check still happens for NVIDIA)
3. Verify `reset_cuda_cache()` resets the new vendor cache variable
4. Verify no circular imports were introduced
5. Run: `cd /home/jacob/Documents/coding/VoiceFlow && uv run -p .venv python -c "from services.gpu import detect_gpu_vendor, get_rocm_status, ROCM_COMPUTE_TYPE; print('imports ok')"` from the `src-pyloid` directory to verify imports work
```

---

## Task 2: Update GpuInfo Data Model and `get_gpu_info()` RPC

**Prerequisites:** Task 1 must be complete

**Prompt:**

```
You are working on the VoiceFlow project. Task 1 (AMD GPU detection in `gpu.py`) is already complete. Your task is to update the `GpuInfo` dataclass and the `get_gpu_info()` RPC method to expose GPU vendor information to the frontend.

## Files to edit
1. `src-pyloid/services/gpu.py` — update `GpuInfo` dataclass
2. `src-pyloid/app_controller.py` — update `get_gpu_info()` method and imports

Read both files first.

## What to implement

### 2a. Update `GpuInfo` dataclass in `gpu.py`

Add three new fields to the existing dataclass:

```python
@dataclass
class GpuInfo:
    cuda_available: bool
    device_count: int
    gpu_name: Optional[str]
    supported_compute_types: list[str]
    current_device: str
    current_compute_type: str
    gpu_vendor: Optional[str]    # NEW: "nvidia", "amd", or None
    rocm_available: bool         # NEW: ROCm libs loadable
    rocm_message: Optional[str]  # NEW: ROCm status message
```

Note: The existing `cudnnAvailable` and `cudnnMessage` fields are NOT on the dataclass — they're only in the dict returned by `get_gpu_info()`. Keep it that way. The new fields follow the same pattern (they'll be in the dict, not necessarily used from the dataclass).

### 2b. Update `get_gpu_info()` in `app_controller.py`

Update the import line to also import `detect_gpu_vendor` and `get_rocm_status` from `services.gpu`.

Update the method:

```python
def get_gpu_info(self) -> dict:
    """Get GPU/CUDA information for the frontend."""
    vendor = detect_gpu_vendor()
    cuda_available = is_cuda_available()
    cudnn_available, cudnn_message = get_cudnn_status()
    rocm_available, rocm_message = get_rocm_status()
    gpu_name = get_gpu_name()
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

## Acceptance criteria
1. `GpuInfo` dataclass has the 3 new fields: `gpu_vendor`, `rocm_available`, `rocm_message`
2. `get_gpu_info()` returns dict with new keys: `gpuVendor`, `rocmAvailable`, `rocmMessage`
3. Import line in `app_controller.py` includes `detect_gpu_vendor` and `get_rocm_status`
4. Existing fields are unchanged — no regressions

## Verification steps
1. Read both files after editing to confirm syntax
2. Run: `cd /home/jacob/Documents/coding/VoiceFlow/src-pyloid && uv run -p ../.venv python -c "from app_controller import AppController; print('import ok')"`
3. Verify the import line has all necessary symbols
```

---

## Task 3: Update Frontend Types and API

**Prerequisites:** Task 2 must be complete (so the backend returns the new fields)

**Prompt:**

```
You are working on the VoiceFlow project. The Python backend now returns three new fields in the GPU info response: `gpuVendor`, `rocmAvailable`, `rocmMessage`. Your task is to update the TypeScript types to match.

## File to edit
`src/lib/types.ts`

Read the file first.

## What to implement

Add three new fields to the `GpuInfo` interface:

```typescript
export interface GpuInfo {
  cudaAvailable: boolean;
  deviceCount: number;
  gpuName: string | null;
  supportedComputeTypes: string[];
  currentDevice: string;
  currentComputeType: string;
  cudnnAvailable: boolean;
  cudnnMessage: string | null;
  gpuVendor: "nvidia" | "amd" | null;  // NEW
  rocmAvailable: boolean;               // NEW
  rocmMessage: string | null;           // NEW
}
```

No changes needed to `src/lib/api.ts` — the existing `getGpuInfo()` call returns whatever the backend sends.

## Acceptance criteria
1. `GpuInfo` interface has `gpuVendor: "nvidia" | "amd" | null`
2. `GpuInfo` interface has `rocmAvailable: boolean`
3. `GpuInfo` interface has `rocmMessage: string | null`
4. All existing fields are unchanged
5. No other files are modified

## Verification steps
1. Read the file after editing to verify the interface is correct
2. Run: `cd /home/jacob/Documents/coding/VoiceFlow && pnpm run lint` to check for TypeScript errors
```

---

## Task 4: Update Settings UI for AMD GPU Status

**Prerequisites:** Task 3 must be complete

**Prompt:**

```
You are working on the VoiceFlow project. The `GpuInfo` TypeScript type now has three new fields: `gpuVendor: "nvidia" | "amd" | null`, `rocmAvailable: boolean`, `rocmMessage: string | null`. Your task is to update the Settings tab to show AMD GPU status alongside NVIDIA.

## File to edit
`src/components/SettingsTab.tsx`

Read the file first. Focus on the "Compute Device" BentoSettingCard (the GPU/Device section in the Advanced area).

## What to change

### 4a. Update the status text display

The current status display (around the `gpuInfo &&` block) shows:
- Green "CUDA Available" when cuda is available
- Amber "cuDNN Missing" when GPU found but no cuDNN
- Gray "CPU Only" otherwise

Update to be vendor-aware:

**Status text logic:**
- If `cudaAvailable`: show "ROCm Available" for AMD, "CUDA Available" for NVIDIA
- If `gpuVendor === "amd"` and `!rocmAvailable`: show "ROCm Setup Needed" (amber)
- If `gpuName` and `!cudnnAvailable` (NVIDIA case): show "cuDNN Missing" (amber)
- Otherwise: "CPU Only" (gray)

**Status color logic:**
- Green if `cudaAvailable`
- Amber if `gpuVendor === "amd"` OR (`gpuName` exists and `!cudnnAvailable`)
- Gray otherwise

### 4b. Update help text for AMD

When AMD GPU is detected but ROCm is not set up, show an AMD-specific message instead of the cuDNN message:

```tsx
{gpuInfo.gpuVendor === "amd" && !gpuInfo.rocmAvailable && (
  <p className="text-xs text-amber-500 pt-1">
    Install ROCm toolkit and ctranslate2-rocm for GPU acceleration.
  </p>
)}
```

### 4c. Show cuDNN message only for NVIDIA

The existing cuDNN missing message (`gpuInfo.gpuName && !gpuInfo.cudnnAvailable`) should only show for NVIDIA GPUs. Add `gpuInfo.gpuVendor !== "amd"` to the condition:

```tsx
{gpuInfo.gpuVendor !== "amd" && gpuInfo.gpuName && !gpuInfo.cudnnAvailable && (
  <p className="text-xs text-amber-500 pt-1">
    Install cuDNN 9.x for GPU acceleration
  </p>
)}
```

### 4d. Update the CUDA dropdown label for AMD

In the device options Select, the CUDA option currently shows `CUDA` or `CUDA (Unavailable)`. Make it vendor-aware:

```tsx
{device === "cuda"
  ? gpuInfo?.gpuVendor === "amd"
    ? `GPU (ROCm)${!gpuInfo?.cudaAvailable ? " (Unavailable)" : ""}`
    : `CUDA${!gpuInfo?.cudaAvailable ? " (Unavailable)" : ""}`
  : device === "auto"
    ? "Auto (Recommended)"
    : "CPU"}
```

## Code style
- Follow the existing code patterns in the file
- Use Tailwind CSS classes consistent with the existing code
- Don't add new imports unless necessary

## Acceptance criteria
1. Status shows "ROCm Available" (green) when AMD GPU is fully configured
2. Status shows "ROCm Setup Needed" (amber) when AMD GPU detected but ROCm missing
3. cuDNN missing message only shows for non-AMD GPUs
4. AMD-specific help text shows when ROCm is not available
5. CUDA dropdown label shows "GPU (ROCm)" for AMD GPUs
6. All existing NVIDIA behavior is preserved

## Verification steps
1. Read the file after editing to verify JSX syntax is correct
2. Run: `cd /home/jacob/Documents/coding/VoiceFlow && pnpm run lint` to check for TypeScript/lint errors
3. Verify no hardcoded "NVIDIA" or "cuDNN" text leaks through when gpuVendor is "amd"
```

---

## Task 5: Update Onboarding UI for AMD GPU

**Prerequisites:** Task 3 must be complete

**Prompt:**

```
You are working on the VoiceFlow project. The `GpuInfo` TypeScript type now has: `gpuVendor: "nvidia" | "amd" | null`, `rocmAvailable: boolean`, `rocmMessage: string | null`. Your task is to update the Onboarding page hardware step to recognize AMD GPUs.

## File to edit
`src/pages/Onboarding.tsx`

Read the file first. Focus on:
1. The `DEVICE_OPTIONS` array (around line 200)
2. The `StepHardware` component
3. The `StepModel` component (also has device references)

## What to change

### 5a. Update `DEVICE_OPTIONS`

The CUDA option currently says "CUDA GPU" with "NVIDIA Only". The StepHardware component receives `gpuInfo` as a prop but `DEVICE_OPTIONS` is defined outside the component as a static array. You need to make the CUDA option labels dynamic based on gpuInfo.

**Approach:** Inside the `StepHardware` component, derive the CUDA option dynamically:

```tsx
const deviceOptions = DEVICE_OPTIONS.map(d => {
  if (d.id === "cuda" && gpuInfo?.gpuVendor === "amd") {
    return {
      ...d,
      label: "GPU (ROCm)",
      desc: "AMD ROCm",
      description: "Uses AMD GPU with ROCm/HIP acceleration. Requires ROCm toolkit and ctranslate2-rocm.",
      bestFor: "Users with AMD GPUs who want GPU-accelerated transcription.",
    };
  }
  return d;
});
```

Then use `deviceOptions` instead of `DEVICE_OPTIONS` in the render.

### 5b. Update GPU status panel (right side details)

In the Hardware Status section:

**Status badge:** Currently shows "Ready"/"Setup Needed"/"CPU Mode". Update:
- "Ready" when `cudaAvailable` (keep as-is)
- "Setup Needed" when `gpuVendor === "amd" && !rocmAvailable` OR `gpuName && !cudnnAvailable`
- "CPU Mode" otherwise

**CUDA/cuDNN labels in the 2-column grid:** When vendor is AMD, change labels:
- Instead of "CUDA" label, show "ROCm"
- Instead of "cuDNN" label, show "HIP Libs"
- The values: for ROCm row, show "Available"/"Unavailable" based on `cudaAvailable`; for HIP Libs row, show "Installed"/"Missing" based on `rocmAvailable`

```tsx
<div className="grid grid-cols-2 gap-2">
  <div className="p-2.5 rounded-lg bg-secondary/30">
    <p className="text-[10px] text-muted-foreground">
      {gpuInfo.gpuVendor === "amd" ? "ROCm" : "CUDA"}
    </p>
    <p className={`text-xs font-medium ${gpuInfo.cudaAvailable ? "text-green-500" : "text-muted-foreground"}`}>
      {gpuInfo.cudaAvailable ? "Available" : "Unavailable"}
    </p>
  </div>
  <div className="p-2.5 rounded-lg bg-secondary/30">
    <p className="text-[10px] text-muted-foreground">
      {gpuInfo.gpuVendor === "amd" ? "HIP Libs" : "cuDNN"}
    </p>
    <p className={`text-xs font-medium ${
      gpuInfo.gpuVendor === "amd"
        ? (gpuInfo.rocmAvailable ? "text-green-500" : "text-amber-500")
        : (gpuInfo.cudnnAvailable ? "text-green-500" : "text-amber-500")
    }`}>
      {gpuInfo.gpuVendor === "amd"
        ? (gpuInfo.rocmAvailable ? "Installed" : "Missing")
        : (gpuInfo.cudnnAvailable ? "Installed" : "Missing")}
    </p>
  </div>
</div>
```

### 5c. Hide cuDNN download section for AMD GPUs

The `showDownloadButton` variable currently shows the download section when `gpuInfo?.gpuName && !gpuInfo?.cudnnAvailable`. Add a vendor check:

```tsx
const showDownloadButton = gpuInfo?.gpuVendor !== "amd" && gpuInfo?.gpuName && !gpuInfo?.cudnnAvailable;
```

### 5d. Add ROCm setup message for AMD

After the cuDNN download section block, add:

```tsx
{gpuInfo?.gpuVendor === "amd" && !gpuInfo?.rocmAvailable && (
  <div className="glass-card p-4 space-y-3">
    <div className="flex items-center gap-2">
      <div className="w-2 h-2 rounded-full bg-amber-500" />
      <span className="text-sm font-medium text-foreground">ROCm Setup Required</span>
    </div>
    <p className="text-xs text-muted-foreground">
      Install ROCm toolkit and ctranslate2-rocm for AMD GPU acceleration. See the ROCm setup guide for instructions.
    </p>
  </div>
)}
```

### 5e. Update bottom status message

The bottom status text currently has 3 states. Update:

```tsx
{gpuInfo?.cudaAvailable
  ? gpuInfo?.gpuVendor === "amd"
    ? "Your AMD GPU is configured for ROCm acceleration."
    : "Your system is fully configured for GPU acceleration."
  : gpuInfo?.gpuVendor === "amd"
    ? "Install ROCm toolkit and ctranslate2-rocm for GPU acceleration."
    : gpuInfo?.gpuName && !gpuInfo?.cudnnAvailable
      ? "Download CUDA libraries from the left panel to enable GPU acceleration."
      : "No compatible GPU detected. CPU transcription works well but is slower."}
```

### 5f. Update StepModel if it has hardcoded CUDA references

Check the `StepModel` component for any hardcoded CUDA references and make them vendor-aware following the same pattern.

## Acceptance criteria
1. CUDA device option shows "GPU (ROCm)" label when AMD GPU detected
2. Hardware status grid shows ROCm/HIP Libs labels for AMD
3. cuDNN download section is hidden for AMD GPUs
4. ROCm setup message appears for AMD GPUs without ROCm
5. Bottom status text is AMD-aware
6. All existing NVIDIA flow preserved

## Verification steps
1. Read the file after editing to verify JSX syntax
2. Run: `cd /home/jacob/Documents/coding/VoiceFlow && pnpm run lint`
3. Search for any remaining hardcoded "NVIDIA" or "cuDNN" text that should be conditional
```

---

## Task 6: Add ROCm Setup Helper Script

**Prerequisites:** None (independent task)

**Prompt:**

```
You are working on the VoiceFlow project. Create a helper script that users can run to set up ROCm for VoiceFlow on Linux.

## File to create
`scripts/setup-rocm.sh`

First, run `ls scripts/` to see if the directory exists. If not, the script will be the first file there.

## Script requirements

Create a bash script with `#!/usr/bin/env bash` and `set -euo pipefail`:

1. **Check for AMD GPU** via `lspci | grep -iq "VGA.*AMD\|Display.*AMD"`
2. **Check ROCm installation** via `command -v rocm-smi`
   - If missing, print install instructions with the official ROCm URL and quick-install apt command
3. **Show detected GPU** via `rocm-smi --showproductname`
4. **Check ROCm libraries** — for each of `libamdhip64.so`, `librocblas.so`, `libhipblas.so`:
   - Check with `ldconfig -p | grep -q "$lib"`
   - Print checkmark or X for each
5. **Build/install ctranslate2 with ROCm** from the community fork:
   - Clone `https://github.com/Yhapcele/CTranslate2-rocm.git` to a temp dir
   - Auto-detect GPU architecture via `rocminfo | grep -oP 'gfx\d+' | head -1` (default to gfx1100)
   - Build with cmake: `-DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" -DCMAKE_BUILD_TYPE=Release`
   - Install Python package via `pip install .`
   - Clean up temp dir
6. Print success message telling user to restart VoiceFlow

## Style
- Use clear section headers with echo
- Print helpful error messages with exit 1 on failure
- Use checkmarks (✓) and X marks (✗) for status
- Make it user-friendly for someone who might not be deeply technical

## Acceptance criteria
1. Script is executable (`#!/usr/bin/env bash`)
2. Uses `set -euo pipefail` for safety
3. Detects AMD GPU via lspci
4. Checks for rocm-smi and provides install instructions if missing
5. Checks all 3 ROCm libraries
6. Auto-detects GPU architecture
7. Builds ctranslate2-rocm from source
8. Cleans up temp directory
9. Has clear error messages at each failure point

## Verification steps
1. Read the file to verify it's syntactically valid bash
2. Run: `bash -n /home/jacob/Documents/coding/VoiceFlow/scripts/setup-rocm.sh` to syntax-check without executing
3. Verify the file has a shebang line and set -euo pipefail
```

---

## Task 7: Update Linux Library Preloading (`main.py`)

**Prerequisites:** Task 1 must be complete

**Prompt:**

```
You are working on the VoiceFlow project. Your task is to extend the Linux library preloading in `main.py` to also handle ROCm libraries alongside the existing NVIDIA preloading.

## File to edit
`src-pyloid/main.py`

Read the file first. Focus on the top section (lines 1-30) where `_preload_nvidia_libs()` is defined and called inside the `if sys.platform.startswith('linux'):` block.

## What to change

### 7a. Rename and restructure the preload function

Rename `_preload_nvidia_libs()` to `_preload_nvidia_pip_libs()` (keeping the same body). Create a new wrapper:

```python
def _preload_gpu_libs():
    """Preload GPU .so libs from pip packages so ctranslate2 can find them."""
    _preload_nvidia_pip_libs()
    _preload_rocm_libs()
```

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

Before the `_preload_gpu_libs()` call, add:

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

### 7d. Update the call site

Change:
```python
try:
    _preload_nvidia_libs()
except Exception:
    pass
```

To:
```python
try:
    _preload_gpu_libs()
except Exception:
    pass
```

## Important constraints
- Everything must stay inside the `if sys.platform.startswith('linux'):` block
- The existing NVIDIA preloading logic must not change (just rename the function)
- Use best-effort pattern (try/except pass) like the existing code
- Don't add logging here — the logger isn't set up yet at this point in the file

## Acceptance criteria
1. `_preload_nvidia_libs` renamed to `_preload_nvidia_pip_libs` (same body)
2. New `_preload_gpu_libs()` calls both NVIDIA and ROCm preload functions
3. `_preload_rocm_libs()` checks `/opt/rocm/lib` and `/opt/rocm/hip/lib`
4. ROCm env vars set before preloading
5. Call site updated from `_preload_nvidia_libs()` to `_preload_gpu_libs()`
6. Everything still inside the Linux platform check
7. Existing NVIDIA preloading behavior unchanged

## Verification steps
1. Read the file to verify the structure
2. Run: `cd /home/jacob/Documents/coding/VoiceFlow/src-pyloid && python -c "import main" 2>&1 | head -5` — this will fail (no Qt) but should NOT show a SyntaxError
3. Verify the function call order: ROCm env vars → `_preload_gpu_libs()` → (nvidia + rocm preloads)
```

---

## Task 8: Update `validate_device_setting()` for AMD

**Prerequisites:** Task 1 must be complete

**Prompt:**

```
You are working on the VoiceFlow project. Task 1 (AMD GPU detection) is already complete in `gpu.py`. Your task is to update `validate_device_setting()` to give AMD-specific error messages.

## File to edit
`src-pyloid/services/gpu.py`

Read the file first. Find the existing `validate_device_setting()` function.

## What to change

Update the function to check the GPU vendor when CUDA is not available and give vendor-specific messages:

```python
def validate_device_setting(device: str) -> tuple[bool, Optional[str]]:
    """Validate that a device setting is usable."""
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
                    return False, "ROCm installed but ctranslate2 lacks ROCm support. Install ctranslate2-rocm or use CPU mode."
                return False, "AMD GPU configuration issue. Check ROCm installation."
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

## Acceptance criteria
1. AMD GPU without ROCm → specific "Install ROCm toolkit" message
2. AMD GPU with ROCm but without ctranslate2-rocm → specific "Install ctranslate2-rocm" message
3. NVIDIA error path preserved exactly as before
4. No GPU → "No compatible GPU detected" message
5. Valid device settings still return `(True, None)`

## Verification steps
1. Read the function to verify all code paths
2. Run: `cd /home/jacob/Documents/coding/VoiceFlow/src-pyloid && uv run -p ../.venv python -c "from services.gpu import validate_device_setting; print(validate_device_setting('auto')); print(validate_device_setting('cpu')); print(validate_device_setting('invalid'))"` to verify basic validation still works
```

---

## Task 9: Add Tests

**Prerequisites:** Tasks 1, 2, and 8 must be complete

**Prompt:**

```
You are working on the VoiceFlow project. AMD GPU detection has been implemented in `gpu.py`. Your task is to write comprehensive unit tests for all AMD GPU detection and status functions.

## File to create
`src-pyloid/tests/test_gpu_amd.py`

First, read these files to understand the code being tested:
- `src-pyloid/services/gpu.py` (the module under test)
- `src-pyloid/tests/test_logger.py` (for test style reference)

## Test structure

Use pytest with `unittest.mock` for mocking. All tests must mock subprocess calls and ctypes.CDLL to avoid requiring actual GPU hardware.

```python
import pytest
from unittest.mock import patch, MagicMock
import subprocess


class TestDetectGpuVendor:
    """Tests for detect_gpu_vendor()."""

    def test_nvidia_gpu_detected(self):
        """nvidia-smi succeeds → returns 'nvidia'"""

    def test_amd_gpu_detected_via_rocm_smi(self):
        """nvidia-smi fails, rocm-smi succeeds → returns 'amd'"""

    def test_amd_gpu_detected_via_lspci(self):
        """nvidia-smi fails, rocm-smi fails, lspci shows AMD → returns 'amd'"""

    def test_no_gpu(self):
        """All detection methods fail → returns None"""

    def test_nvidia_preferred_over_amd(self):
        """Both NVIDIA and AMD present → returns 'nvidia' (checked first)"""


class TestCheckRocmLibsAvailable:
    """Tests for _check_rocm_libs_available()."""

    def test_all_libs_present(self):
        """All ROCm .so files load → (True, None)"""

    def test_missing_lib(self):
        """One .so missing → (False, error_message containing lib name)"""

    def test_all_libs_missing(self):
        """All missing → (False, error about first missing lib)"""


class TestCheckRocmCtranslate2:
    """Tests for _check_rocm_ctranslate2()."""

    def test_rocm_ct2_available(self):
        """ctranslate2 reports CUDA compute types → True"""

    def test_no_cuda_types(self):
        """ctranslate2 returns empty set → False"""

    def test_import_error(self):
        """ctranslate2 not installed → False"""


class TestIsCudaAvailableAmd:
    """Tests for is_cuda_available() with AMD GPU scenarios."""

    def setup_method(self):
        """Reset caches before each test."""
        from services.gpu import reset_cuda_cache
        reset_cuda_cache()

    def test_amd_with_rocm_and_ct2(self):
        """AMD GPU + ROCm libs + ROCm ctranslate2 → True"""

    def test_amd_without_rocm(self):
        """AMD GPU + no ROCm libs → False"""

    def test_amd_with_rocm_no_ct2(self):
        """AMD GPU + ROCm libs + standard ctranslate2 (no CUDA types) → False"""


class TestGetGpuNameAmd:
    """Tests for get_gpu_name() AMD path."""

    def test_amd_gpu_name_from_rocm_smi(self):
        """nvidia-smi fails, rocm-smi returns name → returns GPU name"""

    def test_both_fail(self):
        """Both nvidia-smi and rocm-smi fail → None"""


class TestValidateDeviceAmd:
    """Tests for validate_device_setting() AMD paths."""

    def setup_method(self):
        from services.gpu import reset_cuda_cache
        reset_cuda_cache()

    def test_cuda_with_working_amd(self):
        """AMD fully configured → (True, None)"""

    def test_cuda_amd_no_rocm(self):
        """AMD GPU, ROCm missing → (False, message about ROCm)"""

    def test_cuda_amd_no_ct2(self):
        """AMD GPU, ROCm present, ct2 not ROCm-built → (False, message about ctranslate2)"""


class TestGetRocmStatus:
    """Tests for get_rocm_status()."""

    def test_fully_available(self):
        """ROCm + ct2 → (True, 'ROCm available')"""

    def test_libs_but_no_ct2(self):
        """ROCm libs present, ct2 missing → (False, message about ct2)"""

    def test_no_rocm(self):
        """No ROCm → (False, error message)"""
```

## Mocking patterns

For subprocess mocking:
```python
@patch("services.gpu.subprocess.run")
def test_example(self, mock_run):
    # Simulate nvidia-smi failure
    mock_run.side_effect = FileNotFoundError
    # or
    mock_run.return_value = MagicMock(returncode=1, stdout="")
```

For ctypes.CDLL mocking:
```python
@patch("services.gpu.ctypes.CDLL")
def test_libs(self, mock_cdll):
    mock_cdll.return_value = MagicMock()  # All libs load
    # or
    mock_cdll.side_effect = OSError("not found")  # Lib missing
```

For ctranslate2 mocking:
```python
@patch("services.gpu.ctranslate2", create=True)
# or
with patch.dict("sys.modules", {"ctranslate2": mock_ct2}):
```

## Important notes
- Always call `reset_cuda_cache()` in `setup_method` for tests that use `is_cuda_available()` or `detect_gpu_vendor()` to prevent cache interference between tests
- Mock at the `services.gpu` module level, not at the original module
- Some functions import ctypes/ctranslate2 locally — mock where they're used

## Acceptance criteria
1. At least 20 test cases covering all AMD-related functions
2. All tests use mocks — no real GPU hardware required
3. Tests cover success paths, failure paths, and edge cases
4. Cache reset in setup_method where needed
5. Tests are organized into clear classes

## Verification steps
1. Run: `cd /home/jacob/Documents/coding/VoiceFlow && uv run -p .venv pytest src-pyloid/tests/test_gpu_amd.py -v`
2. All tests must pass
3. Run: `cd /home/jacob/Documents/coding/VoiceFlow && uv run -p .venv pytest src-pyloid/tests/test_gpu_amd.py -v --tb=short` to see any failures with tracebacks
```

---

## Task 10: Update Build Scripts and Dependencies

**Prerequisites:** Tasks 1 and 7 must be complete

**Prompt:**

```
You are working on the VoiceFlow project. AMD GPU support has been added to the Python backend. Your task is to ensure the build configuration doesn't strip ROCm-related libraries.

## Files to check and potentially edit
1. `src-pyloid/build/linux_optimize.spec` — PyInstaller exclusion list
2. `pyproject.toml` — Python dependencies

Read both files first.

## What to do

### 10a. Check `pyproject.toml`

No new pip dependencies should be needed. ROCm libraries come from system packages, not pip. The ctranslate2-rocm build replaces the standard ctranslate2 pip package. Verify that nothing needs to change and document this decision.

### 10b. Update `linux_optimize.spec`

The spec file lists libraries/files to EXCLUDE from the PyInstaller bundle. Review it and make sure:
- No patterns would accidentally exclude ROCm-related `.so` files (libamdhip64, librocblas, libhipblas)
- No patterns would exclude HIP-related files

The spec file currently excludes various Qt and PySide6 libraries. ROCm libs are system-level and won't be bundled by PyInstaller anyway, so likely no changes are needed. But verify this.

If the spec has any broad `*.so` exclusion patterns that could catch ROCm libs, add explicit keep rules.

### 10c. Add a comment

Add a brief comment in `linux_optimize.spec` noting that ROCm libraries are loaded at runtime from system paths (`/opt/rocm/lib`) and are not bundled:

```
# Note: ROCm libraries (libamdhip64.so, librocblas.so, libhipblas.so) are loaded
# at runtime from /opt/rocm/lib and are not bundled with the application.
```

## Acceptance criteria
1. `pyproject.toml` is unchanged (no new dependencies needed)
2. `linux_optimize.spec` doesn't exclude any ROCm-related libraries
3. A comment is added to the spec file explaining ROCm runtime loading
4. No functional changes that could break existing builds

## Verification steps
1. Read both files after any edits
2. Search the spec file for any patterns that could match "rocm", "hip", or "amd" library names
3. Verify no syntax errors in the spec file
```

---

## Task 11: Update CLAUDE.md Documentation

**Prerequisites:** All other tasks should be complete

**Prompt:**

```
You are working on the VoiceFlow project. AMD GPU support via ROCm has been fully implemented. Your task is to update the CLAUDE.md documentation to reflect these changes.

## File to edit
`CLAUDE.md`

Read the file first.

## What to update

### 11a. Update the gpu.py service description

In the Architecture > Backend > Services section, find the `gpu.py` line and update it. Currently it likely just mentions CUDA. Add AMD/ROCm:

Change the description to mention both NVIDIA (CUDA/cuDNN) and AMD (ROCm/HIP) GPU detection.

### 11b. Add a note about device="cuda" working for both vendors

In the "Key Patterns" section or near the transcription flow description, add a note:
- `device="cuda"` works for both NVIDIA and AMD GPUs (AMD via HIP CUDA compatibility layer)
- GPU detection is vendor-aware but the transcription service code is identical for both

### 11c. Update "Linux-Specific" section

Add a bullet point about ROCm:
- **ROCm GPU support**: AMD GPU acceleration via ROCm/HIP. Libraries loaded at runtime from `/opt/rocm/lib`. Setup via `scripts/setup-rocm.sh`

### 11d. Update the Testing section

Add the new test file:
- `test_gpu_amd.py` - AMD GPU detection tests (mocked, no hardware required)

## Important constraints
- Keep changes minimal — only add what's needed for AMD GPU context
- Don't restructure existing content
- Match the existing writing style (terse, technical)
- Don't add a "changelog" or "history" section

## Acceptance criteria
1. gpu.py description mentions both NVIDIA and AMD GPU detection
2. Note about device="cuda" working for both vendors via HIP
3. Linux-Specific section has ROCm bullet point
4. Testing section lists test_gpu_amd.py
5. No unrelated changes to CLAUDE.md

## Verification steps
1. Read the file after editing
2. Verify the changes are minimal and focused
3. Verify no existing content was accidentally removed
```

---

## Summary of task dependencies

```
Task 1 (GPU detection) — no deps
  ├─ Task 2 (Data model) — needs 1
  │   └─ Task 3 (Frontend types) — needs 2
  │       ├─ Task 4 (Settings UI) — needs 3
  │       └─ Task 5 (Onboarding UI) — needs 3
  ├─ Task 7 (Library preloading) — needs 1
  ├─ Task 8 (Validation) — needs 1
  └─ Task 9 (Tests) — needs 1, 2, 8
Task 6 (Setup script) — no deps (independent)
Task 10 (Build scripts) — needs 1, 7
Task 11 (Documentation) — do last
```

**Recommended execution order:** 1 → 6 (parallel) → 2 → 7, 8 (parallel) → 3 → 9 → 4, 5 (parallel) → 10 → 11
