# Task 11: Update CLAUDE.md Documentation

## Objective

Update `CLAUDE.md` at the repository root to document AMD GPU (ROCm) support alongside the existing NVIDIA documentation.

## Context

`CLAUDE.md` is the project documentation file that provides guidance to Claude Code. It describes the architecture, services, patterns, and Linux-specific details. With AMD GPU support now implemented, the documentation needs to reflect the new capabilities.

## File to Modify

- `CLAUDE.md` (repository root)

## Requirements

Read the current `CLAUDE.md` first, then make these specific additions:

### 11a. Update the Architecture > Backend > Services description

In the services list, update the `gpu.py` description. Currently it's not explicitly listed. Add or update:

```
- `gpu.py` - GPU detection and management. Supports NVIDIA (CUDA/cuDNN) and AMD (ROCm/HIP). Detects GPU vendor, checks library availability, validates device settings. Note: `device="cuda"` works for both NVIDIA and AMD (HIP provides CUDA compatibility layer).
```

### 11b. Add AMD GPU info to Recording Flow or Architecture section

Add a brief note somewhere appropriate (near the existing CUDA/device references):

```
### GPU Acceleration

VoiceFlow supports GPU acceleration via:
- **NVIDIA**: CUDA + cuDNN (auto-detected via nvidia-smi and ctranslate2)
- **AMD**: ROCm/HIP (auto-detected via rocm-smi, requires ctranslate2-rocm fork)

Both use `device="cuda"` in faster-whisper/ctranslate2 — HIP provides a CUDA compatibility layer, so the transcription code is vendor-agnostic. GPU vendor detection (`detect_gpu_vendor()`) determines which library checks to run.
```

### 11c. Update Linux-Specific section

Add ROCm details to the existing "Linux-Specific" section:

```
- **ROCm GPU support**: AMD GPUs use ROCm/HIP via a ctranslate2 community fork. Libraries loaded from `/opt/rocm/lib` at runtime. Setup helper: `scripts/setup-rocm.sh`
- **GPU library preloading**: `main.py` preloads both NVIDIA pip-package libs and ROCm system libs at startup via `_preload_gpu_libs()`
```

### 11d. Update Testing section

Add the AMD GPU test file:

```
- `test_gpu_amd.py` - AMD GPU detection, ROCm status, and vendor-aware validation tests (all mocked, no GPU required)
```

### 11e. Add setup-rocm.sh to Commands or a new section

Add under Commands or as a note:

```
# AMD GPU setup (Linux only, requires ROCm toolkit)
bash scripts/setup-rocm.sh
```

## What NOT to Change

- Do NOT rewrite existing sections — only add new content or extend existing descriptions
- Do NOT remove any existing documentation
- Do NOT change the overall structure of the file
- Do NOT add content unrelated to AMD GPU support

## Verification Steps

Before considering this task complete, verify:

1. **Read the updated file**: Confirm all 5 additions (11a through 11e) are present
2. **No removed content**: Compare the file — no existing documentation should be deleted or significantly altered
3. **Accuracy**: Verify technical details are correct:
   - `device="cuda"` works for both NVIDIA and AMD (via HIP)
   - ROCm libs are at `/opt/rocm/lib`
   - Setup script path is `scripts/setup-rocm.sh`
   - Test file is `test_gpu_amd.py`
4. **Formatting**: Ensure Markdown formatting is consistent with the rest of the file (heading levels, bullet styles, code block languages)
5. **No duplicate info**: Verify the same information isn't repeated in multiple sections
