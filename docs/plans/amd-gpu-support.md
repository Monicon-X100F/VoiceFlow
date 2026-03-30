# AMD GPU Support Implementation Plan

## Background

VoiceFlow uses **faster-whisper** (backed by **CTranslate2**) for speech-to-text transcription. GPU acceleration currently only supports NVIDIA GPUs via CUDA/cuDNN. This plan adds AMD GPU support on Linux via ROCm/HIP.

### Key Technical Constraint

CTranslate2 has **no official ROCm pip wheel**. A [community fork](https://github.com/Yhapcele/CTranslate2-rocm) exists that builds CTranslate2 with HIP support. When built with ROCm/HIP, CTranslate2 still uses `device="cuda"` (HIP provides a CUDA compatibility layer), so **faster-whisper's API calls require no changes**.

This means:
- The `TranscriptionService` code stays the same — `device="cuda"` works for both NVIDIA and AMD
- The GPU detection, library checking, and UI need to become vendor-aware
- Users will need ROCm system packages installed, and a ROCm-enabled ctranslate2 build

---

## Architecture Overview

### Current Flow (NVIDIA only)
```
gpu.py: is_cuda_available() → checks ctranslate2 CUDA + cuDNN/cuBLAS libs
        get_gpu_name() → nvidia-smi
        resolve_device() → "auto"/"cpu"/"cuda"
transcription.py: WhisperModel(device="cuda", compute_type="float16")
```

### Target Flow (NVIDIA + AMD)
```
gpu.py: detect_gpu_vendor() → "nvidia" / "amd" / None
        is_gpu_available() → checks CUDA or ROCm libs based on vendor
        get_gpu_name() → nvidia-smi OR rocm-smi
        resolve_device() → still returns "cpu" or "cuda" (HIP uses "cuda" device string)
transcription.py: NO CHANGES (device="cuda" works for both via HIP)
```

---

## Task Breakdown

### Task 1: Add AMD GPU Detection (`gpu.py`)

**File:** `src-pyloid/services/gpu.py`

**What to do:**
Add functions to detect AMD GPUs and check ROCm library availability, alongside the existing NVIDIA detection.

**1a. Add `detect_gpu_vendor()` function**

Detects whether the system has an NVIDIA or AMD GPU. Returns `"nvidia"`, `"amd"`, or `None`.

```python
def detect_gpu_vendor() -> Optional[str]:
    """Detect GPU vendor. Returns 'nvidia', 'amd', or None."""
    # Check NVIDIA first (existing logic - nvidia-smi)
    if _check_nvidia_present():
        return "nvidia"
    # Check AMD via rocm-smi or lspci
    if _check_amd_present():
        return "amd"
    return None
```

Implementation details for `_check_amd_present()`:
- Try running `rocm-smi --showproductname` (subprocess, timeout 5s)
- If `rocm-smi` not found, fall back to checking `lspci | grep -i "VGA.*AMD\|Display.*AMD"` (Linux only)
- Cache the result like `_cuda_available_cache`

**1b. Add `_check_rocm_libs_available()` function**

Similar to `_check_cudnn_available()` but for ROCm libraries:

```python
def _check_rocm_libs_available() -> tuple[bool, Optional[str]]:
    """Check if ROCm/HIP libraries are available for inference."""
    import ctypes
    # Check for key ROCm libraries
    rocm_libs = ["libamdhip64.so", "librocblas.so", "libhipblas.so"]
    for lib_name in rocm_libs:
        try:
            ctypes.CDLL(lib_name)
        except OSError:
            return False, f"ROCm library {lib_name} not found. Install ROCm toolkit."
    return True, None
```

**1c. Add `get_gpu_name()` AMD support**

Extend `get_gpu_name()` to try `rocm-smi` when `nvidia-smi` fails:

```python
# After nvidia-smi fails, try:
result = subprocess.run(
    ["rocm-smi", "--showproductname"],
    capture_output=True, text=True, timeout=5
)
# Parse output for GPU name
```

**1d. Add `_check_rocm_ctranslate2()` function**

Verify the installed ctranslate2 was built with ROCm/HIP support:

```python
def _check_rocm_ctranslate2() -> bool:
    """Check if ctranslate2 is built with ROCm support."""
    try:
        import ctranslate2
        # ROCm-built ctranslate2 reports "cuda" compute types via HIP
        compute_types = ctranslate2.get_supported_compute_types("cuda")
        return len(compute_types) > 0
    except Exception:
        return False
```

**1e. Update `is_cuda_available()` to be vendor-aware**

The function should work for both NVIDIA (cuDNN check) and AMD (ROCm libs check):

```python
def is_cuda_available() -> bool:
    """Check if GPU acceleration is available (NVIDIA CUDA or AMD ROCm/HIP)."""
    # ... existing ctranslate2 check ...

    vendor = detect_gpu_vendor()
    if vendor == "nvidia":
        cudnn_available, _ = _check_cudnn_available()
        return cudnn_available
    elif vendor == "amd":
        rocm_available, _ = _check_rocm_libs_available()
        return rocm_available and _check_rocm_ctranslate2()
    return False
```

**1f. Add constants and update `DEVICE_OPTIONS`**

```python
ROCM_COMPUTE_TYPE = "float16"  # ROCm also supports float16 via HIP
```

No change to `DEVICE_OPTIONS` — device is still `"auto"/"cpu"/"cuda"` since HIP maps to CUDA device string.

**Testing:**
- Add `src-pyloid/tests/test_gpu_amd.py` with unit tests
- Mock `subprocess.run` for `rocm-smi` responses
- Mock `ctypes.CDLL` for ROCm library detection
- Test `detect_gpu_vendor()` returns correct vendor
- Test `is_cuda_available()` with AMD GPU mocks
- Test fallback behavior when ROCm is partially installed

---

### Task 2: Update `GpuInfo` Data Model and `get_gpu_info()` RPC

**Files:** `src-pyloid/services/gpu.py`, `src-pyloid/app_controller.py`

**What to do:**
Extend the `GpuInfo` dataclass and `get_gpu_info()` RPC to expose GPU vendor information to the frontend.

**2a. Update `GpuInfo` dataclass**

```python
@dataclass
class GpuInfo:
    cuda_available: bool
    device_count: int
    gpu_name: Optional[str]
    supported_compute_types: list[str]
    current_device: str
    current_compute_type: str
    cudnn_available: bool       # Keep for NVIDIA
    cudnn_message: Optional[str]
    gpu_vendor: Optional[str]   # NEW: "nvidia", "amd", or None
    rocm_available: bool        # NEW: ROCm libs loadable
    rocm_message: Optional[str] # NEW: ROCm status message
```

**2b. Update `get_gpu_info()` in `app_controller.py`**

```python
def get_gpu_info(self) -> dict:
    vendor = detect_gpu_vendor()
    cuda_available = is_cuda_available()
    cudnn_available, cudnn_message = get_cudnn_status()
    rocm_available, rocm_message = get_rocm_status()  # New function
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
        "gpuVendor": vendor,          # NEW
        "rocmAvailable": rocm_available,  # NEW
        "rocmMessage": rocm_message,      # NEW
    }
```

**2c. Add `get_rocm_status()` function in `gpu.py`**

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

**Testing:**
- Unit test `get_gpu_info()` returns correct dict structure with new fields
- Test with mocked AMD and NVIDIA scenarios

---

### Task 3: Update Frontend Types and API

**Files:** `src/lib/types.ts`, `src/lib/api.ts`

**What to do:**
Add the new GPU vendor fields to TypeScript types.

**3a. Update `GpuInfo` interface in `types.ts`**

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

No changes needed to `api.ts` — the existing `getGpuInfo()` call returns whatever the backend sends.

---

### Task 4: Update Settings UI for AMD GPU Status

**File:** `src/components/SettingsTab.tsx`

**What to do:**
Update the "Compute Device" settings card to show AMD GPU status alongside NVIDIA.

**4a. Update status display logic**

The status indicator currently shows:
- Green: "CUDA Available" (NVIDIA working)
- Amber: "cuDNN Missing" (NVIDIA GPU found, cuDNN not installed)
- Gray: "CPU Only"

Add AMD states:
- Green: "ROCm Available" (AMD working)
- Amber: "ROCm Setup Needed" (AMD GPU found, ROCm libs or ctranslate2-rocm missing)
- Keep existing NVIDIA states

```tsx
// Status text logic
const getStatusText = (gpuInfo: GpuInfo) => {
  if (gpuInfo.cudaAvailable) {
    return gpuInfo.gpuVendor === "amd" ? "ROCm Available" : "CUDA Available";
  }
  if (gpuInfo.gpuVendor === "amd" && !gpuInfo.rocmAvailable) {
    return "ROCm Setup Needed";
  }
  if (gpuInfo.gpuName && !gpuInfo.cudnnAvailable) {
    return "cuDNN Missing";
  }
  return "CPU Only";
};

// Status color logic
const getStatusColor = (gpuInfo: GpuInfo) => {
  if (gpuInfo.cudaAvailable) return "text-green-500";
  if (gpuInfo.gpuVendor === "amd" || (gpuInfo.gpuName && !gpuInfo.cudnnAvailable)) {
    return "text-amber-500";
  }
  return "text-muted-foreground";
};
```

**4b. Update help text for AMD**

When AMD GPU is detected but ROCm is not set up, show:
```tsx
{gpuInfo.gpuVendor === "amd" && !gpuInfo.rocmAvailable && (
  <p className="text-xs text-amber-500 pt-1">
    Install ROCm toolkit and ctranslate2-rocm for GPU acceleration.
    See docs for setup instructions.
  </p>
)}
```

**4c. Hide cuDNN download button for AMD GPUs**

The cuDNN download button should only appear for NVIDIA GPUs:
```tsx
// Only show cuDNN download for NVIDIA GPUs
{gpuInfo.gpuVendor === "nvidia" && gpuInfo.gpuName && !gpuInfo.cudnnAvailable && (
  // ... existing cuDNN download UI ...
)}
```

---

### Task 5: Update Onboarding UI for AMD GPU

**File:** `src/pages/Onboarding.tsx`

**What to do:**
Update the hardware setup step in onboarding to recognize AMD GPUs.

**5a. Update `DEVICE_OPTIONS` descriptions**

Change the "CUDA GPU" option to be more generic, or add vendor-specific descriptions:

```tsx
{
  id: "cuda",
  label: gpuInfo?.gpuVendor === "amd" ? "GPU (ROCm)" : "GPU (CUDA)",
  description: gpuInfo?.gpuVendor === "amd"
    ? "Uses AMD GPU with ROCm/HIP acceleration. Requires ROCm toolkit and ctranslate2-rocm."
    : "Uses NVIDIA GPU with CUDA acceleration. Requires compatible NVIDIA GPU with CUDA libraries.",
}
```

**5b. Update GPU status panel**

- Show "ROCm" status instead of "cuDNN" status when vendor is AMD
- Replace "CUDA" labels with "ROCm" labels for AMD GPUs
- Hide cuDNN download prompt for AMD GPUs
- Show ROCm setup instructions for AMD GPUs

**5c. Update help text**

```tsx
// Bottom info text
{gpuInfo?.gpuVendor === "amd"
  ? gpuInfo?.cudaAvailable
    ? "Your AMD GPU is configured for ROCm acceleration."
    : "Install ROCm toolkit and ctranslate2-rocm for GPU acceleration."
  : /* existing NVIDIA messages */ }
```

---

### Task 6: Add ROCm Setup Helper Script

**File:** `scripts/setup-rocm.sh` (new file)

**What to do:**
Create a helper script that users can run to set up ROCm for VoiceFlow on Linux.

```bash
#!/usr/bin/env bash
# VoiceFlow ROCm Setup Script
# Sets up AMD GPU support for VoiceFlow on Linux

set -euo pipefail

echo "VoiceFlow ROCm Setup"
echo "===================="

# 1. Check for AMD GPU
if ! lspci | grep -iq "VGA.*AMD\|Display.*AMD"; then
    echo "ERROR: No AMD GPU detected."
    exit 1
fi

# 2. Check ROCm installation
if ! command -v rocm-smi &> /dev/null; then
    echo "ROCm toolkit not found."
    echo ""
    echo "Install ROCm following the official guide:"
    echo "  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    echo ""
    echo "Quick install (Ubuntu/Debian):"
    echo "  sudo apt install rocm-hip-sdk rocblas-dev hipblas-dev"
    exit 1
fi

# 3. Show detected GPU
echo "Detected GPU:"
rocm-smi --showproductname 2>/dev/null || echo "  (could not query GPU name)"

# 4. Check ROCm libraries
echo ""
echo "Checking ROCm libraries..."
MISSING=0
for lib in libamdhip64.so librocblas.so libhipblas.so; do
    if ldconfig -p | grep -q "$lib"; then
        echo "  ✓ $lib"
    else
        echo "  ✗ $lib (MISSING)"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "Some ROCm libraries are missing. Install the ROCm toolkit."
    exit 1
fi

# 5. Build/install ctranslate2 with ROCm
echo ""
echo "Installing ctranslate2 with ROCm support..."
echo "This will build from source (requires cmake, python dev headers)."
echo ""

# Clone the ROCm fork
TMPDIR=$(mktemp -d)
git clone https://github.com/Yhapcele/CTranslate2-rocm.git "$TMPDIR/ct2-rocm"
cd "$TMPDIR/ct2-rocm"

# Detect GPU architecture
GFX_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "gfx1100")
echo "Detected GPU architecture: $GFX_ARCH"

# Build
mkdir build && cd build
cmake -DWITH_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make -j$(nproc)

# Install Python package
cd ../python
pip install .

echo ""
echo "ctranslate2-rocm installed successfully!"
echo "Restart VoiceFlow to use GPU acceleration."

# Cleanup
rm -rf "$TMPDIR"
```

---

### Task 7: Update Linux Library Preloading (`main.py`)

**File:** `src-pyloid/main.py`

**What to do:**
Extend `_preload_nvidia_libs()` to also preload ROCm libraries if present.

**7a. Rename and extend the preload function**

```python
def _preload_gpu_libs():
    """Preload GPU .so libs from pip packages so ctranslate2 can find them."""
    import ctypes

    # Existing: preload NVIDIA libs from pip packages
    _preload_nvidia_pip_libs()

    # New: preload ROCm libs if present
    _preload_rocm_libs()

def _preload_nvidia_pip_libs():
    """Existing logic - preload nvidia .so from pip site-packages."""
    # ... existing code from _preload_nvidia_libs() ...

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

**7b. Set ROCm environment variables**

Add before the preload call:
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

---

### Task 8: Update `validate_device_setting()` for AMD

**File:** `src-pyloid/services/gpu.py`

**What to do:**
Update validation to give AMD-specific error messages.

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
            elif vendor == "nvidia":
                # Existing NVIDIA error logic
                ...
            else:
                return False, "No compatible GPU detected. Use CPU mode."
    return True, None
```

---

### Task 9: Add Tests

**File:** `src-pyloid/tests/test_gpu_amd.py` (new file)

**What to do:**
Write comprehensive unit tests for all AMD GPU detection and status functions.

**Test cases:**

```python
class TestDetectGpuVendor:
    def test_nvidia_gpu_detected(self):
        """nvidia-smi succeeds → returns 'nvidia'"""

    def test_amd_gpu_detected_via_rocm_smi(self):
        """rocm-smi succeeds → returns 'amd'"""

    def test_amd_gpu_detected_via_lspci(self):
        """rocm-smi fails, lspci shows AMD → returns 'amd'"""

    def test_no_gpu(self):
        """Both fail → returns None"""

    def test_nvidia_preferred_over_amd(self):
        """Both present → returns 'nvidia' (NVIDIA checked first)"""

class TestRocmLibsAvailable:
    def test_all_libs_present(self):
        """All ROCm .so files load → (True, None)"""

    def test_missing_lib(self):
        """One .so missing → (False, error_message)"""

class TestRocmCtranslate2:
    def test_rocm_ct2_available(self):
        """ctranslate2.get_supported_compute_types('cuda') returns types → True"""

    def test_standard_ct2(self):
        """Standard ctranslate2 without ROCm → False (no CUDA types on AMD system)"""

class TestIsCudaAvailableAmd:
    def test_amd_with_rocm_and_ct2(self):
        """AMD GPU + ROCm libs + ROCm ctranslate2 → True"""

    def test_amd_without_rocm(self):
        """AMD GPU + no ROCm libs → False"""

    def test_amd_with_rocm_no_ct2(self):
        """AMD GPU + ROCm libs + standard ctranslate2 → False"""

class TestGetGpuName:
    def test_amd_gpu_name_from_rocm_smi(self):
        """rocm-smi returns GPU name correctly"""

    def test_fallback_to_none(self):
        """Both nvidia-smi and rocm-smi fail → None"""

class TestValidateDeviceAmd:
    def test_cuda_with_working_amd(self):
        """AMD fully configured → (True, None)"""

    def test_cuda_with_partial_amd(self):
        """AMD GPU found, ROCm missing → (False, helpful_message)"""

class TestGetRocmStatus:
    def test_rocm_fully_available(self):
        """Returns (True, 'ROCm available')"""

    def test_rocm_libs_but_no_ct2(self):
        """Returns (False, 'ctranslate2 not built with ROCm support')"""

    def test_no_rocm(self):
        """Returns (False, error_message)"""
```

All tests should mock subprocess calls and ctypes.CDLL to avoid requiring actual GPU hardware.

---

### Task 10: Update Build Scripts and Dependencies

**Files:** `pyproject.toml`, `src-pyloid/build/linux_optimize.spec`

**What to do:**

**10a. No new pip dependencies needed**

ROCm libraries come from system packages, not pip. The ctranslate2-rocm build replaces the standard ctranslate2 pip package. No changes to `pyproject.toml` dependencies.

**10b. Update Linux build spec**

Ensure the PyInstaller spec doesn't exclude ROCm-related `.so` files if they get bundled:

```python
# In linux_optimize.spec, don't strip ROCm libs if present
```

**10c. Add optional ROCm note to setup**

In `package.json` setup script or a setup doc, note that AMD GPU users should run `scripts/setup-rocm.sh` after the standard setup.

---

### Task 11: Update `CLAUDE.md` Documentation

**File:** `CLAUDE.md`

**What to do:**
Add AMD GPU information to the relevant sections.

- Add "AMD GPU (ROCm)" to the Architecture > Services > gpu.py description
- Note that `device="cuda"` works for both NVIDIA and AMD (via HIP)
- Add ROCm setup instructions reference
- Update "Linux-Specific" section with ROCm details

---

## Task Dependency Graph

```
Task 1 (GPU detection)
  ├── Task 2 (Data model + RPC) ── depends on Task 1
  │     ├── Task 3 (Frontend types) ── depends on Task 2
  │     │     ├── Task 4 (Settings UI) ── depends on Task 3
  │     │     └── Task 5 (Onboarding UI) ── depends on Task 3
  │     └── Task 8 (Validation) ── depends on Task 2
  ├── Task 7 (Library preloading) ── depends on Task 1
  └── Task 9 (Tests) ── depends on Tasks 1, 2, 8
Task 6 (Setup script) ── independent
Task 10 (Build/deps) ── depends on Tasks 1, 7
Task 11 (Documentation) ── do last
```

## Recommended Execution Order

1. **Task 1** — Core GPU detection (foundation for everything)
2. **Task 7** — Library preloading (pairs with detection)
3. **Task 2** — Data model and RPC updates
4. **Task 8** — Validation updates
5. **Task 9** — Tests (validate Tasks 1-2-7-8 work)
6. **Task 3** — Frontend types
7. **Task 4** — Settings UI
8. **Task 5** — Onboarding UI
9. **Task 6** — Setup script
10. **Task 10** — Build/packaging
11. **Task 11** — Documentation

## Testing Strategy

- **Unit tests** (Task 9): All GPU detection functions with mocked subprocess/ctypes — can run on any machine
- **Manual testing on AMD hardware**: Run VoiceFlow on a system with AMD GPU + ROCm installed, verify detection, transcription, and UI
- **Regression testing on NVIDIA**: Ensure existing NVIDIA flow is not broken — run existing tests
- **CPU fallback testing**: Verify "auto" mode falls back to CPU when neither GPU vendor is fully configured

## Risk Considerations

1. **CTranslate2 ROCm fork maintenance**: The fork (`Yhapcele/CTranslate2-rocm`) is community-maintained. If it goes stale, AMD support breaks. Mitigate by abstracting detection so a future backend swap (e.g., whisper.cpp with Vulkan) is easier.

2. **GPU architecture compatibility**: ROCm requires matching `gfx` architecture codes. The setup script auto-detects via `rocminfo`, but some GPUs may need `HSA_OVERRIDE_GFX_VERSION` workarounds.

3. **No automated ctranslate2-rocm distribution**: Unlike NVIDIA (pip install), AMD users must build from source. The setup script (Task 6) helps, but it's a higher barrier.

4. **Dual-GPU systems**: Systems with both NVIDIA and AMD GPUs will prefer NVIDIA (checked first in `detect_gpu_vendor()`). This is intentional since NVIDIA has better out-of-box support.
