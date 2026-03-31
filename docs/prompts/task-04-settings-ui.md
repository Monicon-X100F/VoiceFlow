# Task 4: Update Settings UI for AMD GPU Status

## Objective

Update the "Compute Device" settings card in `SettingsTab.tsx` to show AMD GPU status (ROCm) alongside the existing NVIDIA (CUDA/cuDNN) status.

## Context

The Settings tab has a "Compute Device" card that shows GPU status. Currently it only handles NVIDIA scenarios:
- Green status: "CUDA Available" (NVIDIA working)
- Amber status: "cuDNN Missing" (NVIDIA GPU found, cuDNN not installed) with a download button
- Gray status: "CPU Only" (no GPU)

The backend now returns `gpuVendor` (`"nvidia" | "amd" | null`), `rocmAvailable` (boolean), and `rocmMessage` (string) in the GPU info. The `GpuInfo` TypeScript interface has been updated (Task 3).

## File to Modify

- `src/components/SettingsTab.tsx`

## Requirements

### 4a. Update status display logic

Read the file first to find the exact location of GPU status rendering. Then update the status text and color logic to handle AMD:

**Status text:**
- If `gpuInfo.cudaAvailable` and vendor is `"amd"` → "ROCm Available"
- If `gpuInfo.cudaAvailable` and vendor is `"nvidia"` (or null) → "CUDA Available" (existing)
- If vendor is `"amd"` and `!rocmAvailable` → "ROCm Setup Needed"
- If vendor is `"nvidia"` and `gpuName` exists and `!cudnnAvailable` → "cuDNN Missing" (existing)
- Otherwise → "CPU Only" (existing)

**Status color:**
- Green (`text-green-500` or equivalent): `cudaAvailable` is true
- Amber (`text-amber-500` or equivalent): AMD GPU without ROCm, OR NVIDIA GPU without cuDNN
- Gray (`text-muted-foreground` or equivalent): no GPU / CPU only

### 4b. Add AMD-specific help text

When an AMD GPU is detected but ROCm is not set up, show a help message:
```
Install ROCm toolkit and ctranslate2-rocm for GPU acceleration.
```

Use a similar style to the existing cuDNN missing message (small text, amber color).

### 4c. Hide cuDNN download button for AMD GPUs

The existing cuDNN download button/UI should only appear when `gpuVendor === "nvidia"` (or `gpuVendor === null` for backward compatibility with systems that haven't updated the backend yet). Never show the cuDNN download option for AMD GPUs.

### 4d. Show GPU vendor in the GPU name display

If the GPU name is shown (e.g., "NVIDIA GeForce RTX 4090"), keep showing it as-is. The `gpuName` from `rocm-smi` will already contain the AMD GPU name. No special formatting needed — just ensure the name displays correctly regardless of vendor.

## What NOT to Change

- Do NOT modify any Python backend files
- Do NOT modify the `types.ts` or `api.ts` files
- Do NOT change the device selection dropdown behavior (auto/cpu/cuda options stay the same)
- Do NOT add new components or files — all changes go in `SettingsTab.tsx`
- Do NOT change styling of existing NVIDIA-related UI elements

## Verification Steps

Before considering this task complete, verify:

1. **TypeScript compiles**: Run `cd /home/user/VoiceFlow && pnpm run lint` — must pass with no errors
2. **Read the modified file**: Verify the status logic handles all 5 scenarios: NVIDIA working, AMD working, AMD partial, NVIDIA partial (cuDNN missing), CPU only
3. **cuDNN download guarded**: Search for the cuDNN download button/section and verify it's wrapped in a condition that excludes AMD GPUs
4. **ROCm help text present**: Verify there's a conditional block that shows ROCm setup guidance when `gpuVendor === "amd" && !rocmAvailable`
5. **No hardcoded "CUDA Available" without vendor check**: Search for any remaining hardcoded "CUDA Available" text that doesn't consider the vendor — it should say "ROCm Available" for AMD
