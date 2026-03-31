# Task 9: Add Comprehensive AMD GPU Tests

## Objective

Create a comprehensive test suite in `src-pyloid/tests/test_gpu_amd.py` covering all AMD GPU detection, status, and validation functions. All tests must use mocks — no real GPU hardware required.

## Context

Tasks 1, 2, and 8 added AMD GPU support to `gpu.py` and `app_controller.py`. This task creates (or completes, if partially created by earlier tasks) a thorough test file covering all the new functionality.

The test file should be at `src-pyloid/tests/test_gpu_amd.py`. If this file already exists from earlier tasks, extend it to cover all cases listed below. If it doesn't exist, create it from scratch.

## File to Create/Modify

- `src-pyloid/tests/test_gpu_amd.py`

## Existing Test Patterns

Look at existing test files in `src-pyloid/tests/` for patterns (e.g., `test_logger.py`, `test_model_manager.py`). Tests use pytest. The working directory for tests is `VoiceFlow/` and imports use `from services.gpu import ...`.

## Required Test Classes and Cases

### `TestDetectGpuVendor`

All tests mock `subprocess.run`. Reset any vendor detection cache before each test.

1. **`test_nvidia_gpu_detected`** — nvidia-smi returns 0 → `"nvidia"`
2. **`test_amd_gpu_detected_via_rocm_smi`** — nvidia-smi fails (FileNotFoundError), rocm-smi returns 0 → `"amd"`
3. **`test_amd_gpu_detected_via_lspci`** — nvidia-smi fails, rocm-smi fails, lspci output contains "VGA.*AMD" → `"amd"`
4. **`test_no_gpu`** — all detection methods fail → `None`
5. **`test_nvidia_preferred_over_amd`** — nvidia-smi succeeds (don't even check AMD) → `"nvidia"`

### `TestRocmLibsAvailable`

Mock `ctypes.CDLL` to simulate library loading.

6. **`test_all_libs_present`** — all 3 ROCm libs load → `(True, None)`
7. **`test_missing_hip_lib`** — `libamdhip64.so` raises OSError → `(False, "...libamdhip64.so not found...")`
8. **`test_missing_rocblas`** — `librocblas.so` raises OSError → `(False, "...librocblas.so not found...")`

### `TestRocmCtranslate2`

Mock `ctranslate2.get_supported_compute_types`.

9. **`test_rocm_ct2_available`** — returns `{"float16", "int8"}` → `True`
10. **`test_standard_ct2_no_cuda`** — raises RuntimeError → `False`
11. **`test_import_error`** — ctranslate2 import fails → `False`

### `TestIsCudaAvailableAmd`

Mock vendor detection, ROCm libs check, ROCm CT2 check. Reset `_cuda_available_cache` before each test.

12. **`test_amd_full_setup`** — AMD vendor + ROCm libs OK + ROCm CT2 OK → `True`
13. **`test_amd_no_rocm_libs`** — AMD vendor + ROCm libs fail → `False`
14. **`test_amd_no_rocm_ct2`** — AMD vendor + ROCm libs OK + standard CT2 → `False`
15. **`test_nvidia_still_works`** — NVIDIA vendor + cuDNN OK → `True` (regression check)
16. **`test_no_gpu_returns_false`** — no ctranslate2 CUDA support → `False`

### `TestGetGpuNameAmd`

Mock `subprocess.run`.

17. **`test_amd_name_from_rocm_smi`** — nvidia-smi fails, rocm-smi returns product name → correct name string
18. **`test_both_fail_returns_none`** — both nvidia-smi and rocm-smi fail → `None`
19. **`test_nvidia_name_still_works`** — nvidia-smi returns name → correct name (regression)

### `TestValidateDeviceAmd`

Mock `is_cuda_available`, `detect_gpu_vendor`, `_check_rocm_libs_available`, `_check_rocm_ctranslate2`.

20. **`test_cuda_amd_working`** — AMD fully working → `(True, None)`
21. **`test_cuda_amd_no_rocm`** — AMD, no ROCm → `(False, "...ROCm is not installed...")`
22. **`test_cuda_amd_no_ct2`** — AMD, ROCm OK, wrong CT2 → `(False, "...ctranslate2 lacks ROCm...")`
23. **`test_cuda_nvidia_no_cudnn`** — NVIDIA, no cuDNN → `(False, "...cuDNN...")`
24. **`test_cuda_no_gpu`** — no vendor → `(False, "No compatible GPU...")`
25. **`test_cpu_always_valid`** — `"cpu"` → `(True, None)`
26. **`test_auto_always_valid`** — `"auto"` → `(True, None)`
27. **`test_invalid_device`** — `"tpu"` → `(False, "Invalid device option...")`

### `TestGetRocmStatus`

Mock `_check_rocm_libs_available` and `_check_rocm_ctranslate2`.

28. **`test_fully_available`** — libs OK + CT2 OK → `(True, "ROCm available")`
29. **`test_libs_but_no_ct2`** — libs OK + CT2 not built → `(False, "...ctranslate2 not built with ROCm...")`
30. **`test_no_rocm`** — libs fail → `(False, error_message)`

## Implementation Notes

- Use `pytest` fixtures for common mock setups
- Use `unittest.mock.patch` for mocking
- Reset `_cuda_available_cache` (and any vendor cache) in each test or use a fixture:
  ```python
  @pytest.fixture(autouse=True)
  def reset_caches():
      import services.gpu as gpu_module
      gpu_module._cuda_available_cache = None
      # Reset vendor cache if it exists
      yield
      gpu_module._cuda_available_cache = None
  ```
- Use `side_effect` on `subprocess.run` mock to simulate different behaviors for different commands
- For `ctypes.CDLL` mocking, use `side_effect` to raise `OSError` for specific library names

## What NOT to Change

- Do NOT modify any source files (`gpu.py`, `app_controller.py`, etc.)
- Do NOT modify existing test files
- Do NOT add test dependencies to `pyproject.toml` (pytest and unittest.mock are already available)

## Verification Steps

Before considering this task complete, verify:

1. **All tests pass**: Run `cd VoiceFlow && uv run -p .venv pytest src-pyloid/tests/test_gpu_amd.py -v` — all 30 tests must pass
2. **No test isolation issues**: Run the tests twice in a row — results must be identical (caches properly reset)
3. **Existing tests unaffected**: Run `cd VoiceFlow && uv run -p .venv pytest src-pyloid/tests/ -v --ignore=src-pyloid/tests/test_transcription.py` — no regressions
4. **Test count**: Run `cd VoiceFlow && uv run -p .venv pytest src-pyloid/tests/test_gpu_amd.py --collect-only` — should show at least 28 test items
5. **No real subprocess calls**: Grep the test file for any unguarded `subprocess.run` calls — all should be mocked
