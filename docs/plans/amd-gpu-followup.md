# AMD GPU Support — Follow-up Revisions

Post-implementation review of the AMD GPU support plan. All 11 original tasks were completed and tests pass (34/34). This plan covers inconsistencies and minor gaps found during review.

---

## Task 1: Make Settings UI CUDA dropdown label AMD-aware

**File:** `src/components/SettingsTab.tsx`

**Problem:** In the Compute Device card, the CUDA option always displays as `CUDA` or `CUDA (Unavailable)` regardless of GPU vendor. Meanwhile, Onboarding correctly shows "GPU (ROCm)" for AMD users. This inconsistency can confuse AMD users who see "CUDA" in Settings but "GPU (ROCm)" in Onboarding.

**What to do:**

Update the CUDA `SelectItem` label (~line 624) to be vendor-aware:

```tsx
{device === "auto"
  ? "Auto (Recommended)"
  : device === "cuda"
    ? gpuInfo?.gpuVendor === "amd"
      ? `GPU (ROCm)${!gpuInfo?.cudaAvailable ? " (Unavailable)" : ""}`
      : `CUDA${!gpuInfo?.cudaAvailable ? " (Unavailable)" : ""}`
    : "CPU"}
```

This makes Settings consistent with Onboarding.

---

## Task 2: Guard cuDNN download button visibility in Settings for AMD

**File:** `src/components/SettingsTab.tsx`

**Problem:** The plan (Task 4c) specified hiding the cuDNN download button for AMD GPUs. The current Settings card doesn't have a standalone cuDNN download button in the Compute Device section (cuDNN download is only in Onboarding and the Danger Zone cleanup). However, the amber help text at line 675 uses `gpuInfo.gpuVendor !== "amd"` which is correct. Verify no cuDNN download prompt leaks through for AMD users.

**What to do:**

Verify that the cuDNN download-related UI (`CudnnDownloadInfo`, download button, progress) in Settings is only reachable for NVIDIA users. If there are other locations where cuDNN download can be triggered (e.g., a separate modal or button added elsewhere), add `gpuInfo.gpuVendor !== "amd"` guards.

If no such leak exists, this task is a no-op — mark as verified.

---

## Task 3: Add `get_gpu_count()` AMD support

**File:** `src-pyloid/services/gpu.py`

**Problem:** `get_gpu_count()` only queries `nvidia-smi` for device count. With AMD, it silently falls back to `return 1`. This is functional but could miscount on multi-AMD-GPU systems.

**What to do:**

After the nvidia-smi attempt fails, try `rocm-smi --showid` or parse `rocminfo` to count AMD GPUs:

```python
def get_gpu_count() -> int:
    if not is_cuda_available():
        return 0

    # Try NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=creationflags
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n'))
    except Exception:
        pass

    # Try AMD via rocm-smi
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            # Count lines that start with GPU index (e.g., "GPU[0]")
            count = sum(1 for line in result.stdout.splitlines() if line.strip().startswith("GPU"))
            if count > 0:
                return count
    except Exception:
        pass

    return 1
```

**Priority:** Low — single-GPU AMD systems work fine already. Only affects display accuracy on multi-GPU setups.

---

## Dependency Graph

```
Task 1 (Settings label) — independent
Task 2 (cuDNN guard verification) — independent
Task 3 (GPU count AMD) — independent
```

All tasks are independent and can be done in any order.

## Priority

These are all minor polish items. None block AMD GPU functionality. Task 1 (label consistency) is the most user-visible.
