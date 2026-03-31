# Task 10: Update Build Scripts and Dependencies

## Objective

Ensure the build and packaging configuration doesn't interfere with AMD ROCm support. Update the Linux build spec to not exclude ROCm libraries and add a note about AMD GPU setup.

## Context

VoiceFlow uses PyInstaller for building desktop apps. The Linux build spec is at `src-pyloid/build/linux_optimize.spec`. ROCm libraries come from system packages (not pip), so no new pip dependencies are needed. The main concern is that PyInstaller's analysis or exclusion lists don't strip ROCm-related files.

## Files to Modify

- `src-pyloid/build/linux_optimize.spec` — ensure ROCm libs aren't excluded
- `pyproject.toml` — no dependency changes needed, but verify and add a comment if appropriate

## Requirements

### 10a. Check and update Linux build spec

Read `src-pyloid/build/linux_optimize.spec` carefully. Look for:

1. **Exclusion lists** (`excludes=`, `exclude_datas=`, etc.) — ensure nothing would exclude `rocm`, `hip`, `amdhip`, `rocblas`, or `hipblas` related files
2. **Binary collection** — if there's custom binary collection logic, ensure it doesn't filter out non-NVIDIA GPU libraries
3. **Hidden imports** — no new hidden imports are needed for ROCm (it's system-level, not pip)

If the spec has exclusion patterns that could catch ROCm files, add exceptions. If it's clean, add a comment noting that ROCm system libraries are intentionally not bundled (they come from the user's system):

```python
# Note: ROCm/HIP libraries for AMD GPU support are loaded from the system
# (/opt/rocm/lib) at runtime and are not bundled with the application.
```

### 10b. Verify pyproject.toml

Read `pyproject.toml` and confirm:
- No new pip dependencies are needed for AMD support
- The existing ctranslate2 dependency doesn't pin to a NVIDIA-specific version that would conflict with the ROCm fork

If ctranslate2 is listed as a dependency, note that AMD users will replace it with the ROCm fork build. This doesn't require a code change — just verify there's no version pin that would cause issues.

### 10c. Add AMD setup note

If there's a setup section in `package.json` scripts or a setup doc, check if it's appropriate to add a brief note. If `package.json` has a `"setup"` script, don't modify it — AMD setup is a separate manual step via `scripts/setup-rocm.sh`.

## What NOT to Change

- Do NOT add new pip dependencies
- Do NOT modify the `package.json` setup script
- Do NOT change any Python source files
- Do NOT change the Windows or macOS build specs
- Do NOT bundle ROCm libraries into the build (they must come from the user's system)

## Verification Steps

Before considering this task complete, verify:

1. **Read the Linux spec**: Confirm no exclusion patterns would catch ROCm libraries
2. **Comment present**: A comment explaining ROCm libraries are loaded at runtime from the system is present in the Linux spec
3. **pyproject.toml checked**: Confirm ctranslate2 dependency doesn't have a version constraint that would prevent the ROCm fork
4. **No new dependencies added**: `git diff pyproject.toml` shows no dependency additions (only comments if anything)
5. **Build spec syntax valid**: The spec file should still be valid Python — run `cd VoiceFlow && python -m py_compile src-pyloid/build/linux_optimize.spec` or at least verify it parses as Python
