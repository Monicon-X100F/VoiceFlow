# Task 6: Add ROCm Setup Helper Script

## Objective

Create a helper shell script (`scripts/setup-rocm.sh`) that Linux users with AMD GPUs can run to set up ROCm support for VoiceFlow. This task is fully independent of all other tasks.

## Context

Unlike NVIDIA (where cuDNN can be pip-installed or downloaded), AMD GPU support requires:
1. System-level ROCm toolkit installation
2. Building ctranslate2 from a community ROCm fork (https://github.com/Yhapcele/CTranslate2-rocm)

This script guides users through the process and automates what it can.

## File to Create

- `scripts/setup-rocm.sh` (new file)

## Requirements

Create a bash script with `set -euo pipefail` that performs these steps:

### Step 1: Check for AMD GPU
- Use `lspci | grep -iq "VGA.*AMD\|Display.*AMD"` to detect AMD GPU
- Exit with clear error message if no AMD GPU found

### Step 2: Check ROCm installation
- Check if `rocm-smi` is available via `command -v`
- If not found, print installation instructions with the official ROCm docs URL: `https://rocm.docs.amd.com/projects/install-on-linux/en/latest/`
- Include quick install hint for Ubuntu/Debian: `sudo apt install rocm-hip-sdk rocblas-dev hipblas-dev`
- Exit with error if ROCm not installed

### Step 3: Show detected GPU
- Run `rocm-smi --showproductname` to display the GPU name

### Step 4: Check ROCm libraries
- Check for `libamdhip64.so`, `librocblas.so`, `libhipblas.so` using `ldconfig -p`
- Print checkmark or X for each library
- Exit with error if any are missing

### Step 5: Build and install ctranslate2 with ROCm
- Print a notice that this builds from source and requires cmake + python dev headers
- Clone the ROCm fork to a temp directory
- Auto-detect GPU architecture via `rocminfo | grep -oP 'gfx\d+'` (default to `gfx1100` if detection fails)
- Build with cmake: `-DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" -DCMAKE_BUILD_TYPE=Release`
- Run `make -j$(nproc)`
- Install the Python package from the `python/` subdirectory
- Clean up the temp directory
- Print success message telling user to restart VoiceFlow

### Script header
- Shebang: `#!/usr/bin/env bash`
- Include a comment header: "VoiceFlow ROCm Setup Script"
- Use `echo` for all output with clear section headers

## What NOT to Change

- Do NOT modify any existing files
- Do NOT add this script to any build process or package.json

## Verification Steps

Before considering this task complete, verify:

1. **File exists**: `scripts/setup-rocm.sh` exists
2. **Executable**: The file should have a proper shebang line (`#!/usr/bin/env bash`)
3. **Shell syntax valid**: Run `bash -n /home/user/VoiceFlow/scripts/setup-rocm.sh` — must pass with no syntax errors
4. **Error handling**: Verify `set -euo pipefail` is present near the top
5. **All 5 steps present**: Read through the script and confirm it has AMD detection, ROCm check, GPU display, library check, and ctranslate2 build sections
6. **Cleanup**: Verify the temp directory is cleaned up at the end (using `rm -rf "$TMPDIR"`)
7. **No hardcoded paths**: Verify the script doesn't hardcode any user-specific paths (home dirs, etc.)
