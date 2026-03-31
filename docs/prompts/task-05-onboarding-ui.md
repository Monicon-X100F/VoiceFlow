# Task 5: Update Onboarding UI for AMD GPU

## Objective

Update the hardware setup step in `Onboarding.tsx` to recognize AMD GPUs and show appropriate ROCm status instead of NVIDIA-specific CUDA/cuDNN messaging.

## Context

The onboarding flow has a hardware configuration step where users select their compute device (auto/cpu/cuda). Currently all GPU messaging is NVIDIA-specific: labels say "CUDA", status shows cuDNN availability, and help text references NVIDIA drivers.

The `GpuInfo` type now includes `gpuVendor` (`"nvidia" | "amd" | null`), `rocmAvailable`, and `rocmMessage`.

## File to Modify

- `src/pages/Onboarding.tsx`

## Requirements

Read the file first to understand the current structure and find the hardware setup step.

### 5a. Update device option labels

Where device options are displayed (the "cuda" option), make the label vendor-aware:
- If `gpuInfo?.gpuVendor === "amd"` → label: "GPU (ROCm)", description references AMD/ROCm
- Otherwise → keep existing "GPU (CUDA)" label and NVIDIA description

### 5b. Update GPU status panel

In the hardware step's GPU status display:
- When vendor is `"amd"`:
  - Show "ROCm" status labels instead of "cuDNN" labels
  - Show `rocmMessage` instead of `cudnnMessage`
  - Use the same green/amber/gray color logic as Task 4
- When vendor is `"nvidia"` or `null`:
  - Keep all existing NVIDIA/cuDNN UI unchanged

### 5c. Hide cuDNN download prompt for AMD

If there's a cuDNN download prompt or button in the onboarding flow, gate it behind `gpuVendor !== "amd"`. AMD users don't need cuDNN.

### 5d. Update help/info text

The bottom info text or helper messages should be vendor-aware:
- AMD with working ROCm: "Your AMD GPU is configured for ROCm acceleration."
- AMD without ROCm: "Install ROCm toolkit and ctranslate2-rocm for GPU acceleration."
- NVIDIA scenarios: keep existing messages unchanged

## What NOT to Change

- Do NOT modify any Python backend files
- Do NOT modify `types.ts` or `api.ts`
- Do NOT change the onboarding flow structure or step order
- Do NOT change the model download step
- Do NOT modify any other pages or components

## Verification Steps

Before considering this task complete, verify:

1. **TypeScript compiles**: Run `cd /home/user/VoiceFlow && pnpm run lint` — must pass with no errors
2. **Read the modified file**: Confirm the hardware step handles both AMD and NVIDIA vendors
3. **Vendor-aware labels**: Search for "CUDA" in the file — every instance should either be in a NVIDIA-specific conditional branch or be a device string literal (`"cuda"`) that's shared between vendors
4. **cuDNN download gated**: If a cuDNN download section exists in onboarding, verify it checks `gpuVendor !== "amd"`
5. **ROCm messaging exists**: Verify there are conditional blocks showing ROCm-specific text when vendor is `"amd"`
6. **No broken JSX**: Verify all conditional rendering uses proper JSX syntax (no dangling `&&` or ternary expressions)
