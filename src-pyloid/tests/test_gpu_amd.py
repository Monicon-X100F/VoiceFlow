"""Tests for AMD GPU detection in gpu.py."""
import sys
import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import gpu.py directly to avoid triggering services/__init__.py
# which pulls in numpy/sounddevice/etc. via AudioService
_gpu_path = Path(__file__).parent.parent / "services" / "gpu.py"
_spec = importlib.util.spec_from_file_location("services.gpu", _gpu_path)
gpu_module = importlib.util.module_from_spec(_spec)
# Stub the logger so gpu.py doesn't need the full services package
sys.modules.setdefault("services.logger", MagicMock(get_logger=lambda _: MagicMock()))
_spec.loader.exec_module(gpu_module)

detect_gpu_vendor = gpu_module.detect_gpu_vendor
_check_rocm_libs_available = gpu_module._check_rocm_libs_available
_check_rocm_ctranslate2 = gpu_module._check_rocm_ctranslate2
get_gpu_name = gpu_module.get_gpu_name
is_cuda_available = gpu_module.is_cuda_available
get_rocm_status = gpu_module.get_rocm_status
validate_device_setting = gpu_module.validate_device_setting
GpuInfo = gpu_module.GpuInfo


def _reset_caches():
    gpu_module._cuda_available_cache = None
    gpu_module._gpu_vendor_cache = "unset"


# ---------------------------------------------------------------------------
# TestDetectGpuVendor
# ---------------------------------------------------------------------------

class TestDetectGpuVendor:
    def setup_method(self):
        _reset_caches()

    def test_nvidia_gpu_detected(self):
        def fake_run(args, **kwargs):
            if args[0] == "nvidia-smi":
                r = MagicMock()
                r.returncode = 0
                r.stdout = ""
                return r
            raise FileNotFoundError

        with patch("subprocess.run", side_effect=fake_run):
            assert detect_gpu_vendor() == "nvidia"

    def test_amd_gpu_detected_via_rocm_smi(self):
        def fake_run(args, **kwargs):
            if args[0] == "nvidia-smi":
                raise FileNotFoundError
            if args[0] == "rocm-smi":
                r = MagicMock()
                r.returncode = 0
                r.stdout = "Card series: Radeon RX 7900 XTX"
                return r
            raise FileNotFoundError

        with patch("subprocess.run", side_effect=fake_run):
            assert detect_gpu_vendor() == "amd"

    def test_amd_gpu_detected_via_lspci(self):
        def fake_run(args, **kwargs):
            if args[0] == "nvidia-smi":
                raise FileNotFoundError
            if args[0] == "rocm-smi":
                raise FileNotFoundError
            if args[0] == "lspci":
                r = MagicMock()
                r.returncode = 0
                r.stdout = "00:02.0 VGA compatible controller: Advanced Micro Devices [AMD] Radeon"
                return r
            if args[0] == "grep":
                r = MagicMock()
                r.returncode = 0
                r.stdout = "00:02.0 VGA compatible controller: Advanced Micro Devices [AMD] Radeon\n"
                return r
            raise FileNotFoundError

        with patch("subprocess.run", side_effect=fake_run):
            assert detect_gpu_vendor() == "amd"

    def test_no_gpu(self):
        def fake_run(args, **kwargs):
            raise FileNotFoundError

        with patch("subprocess.run", side_effect=fake_run):
            assert detect_gpu_vendor() is None

    def test_nvidia_preferred_over_amd(self):
        """nvidia-smi succeeds → returns nvidia even if AMD tools would also succeed."""
        def fake_run(args, **kwargs):
            if args[0] == "nvidia-smi":
                r = MagicMock()
                r.returncode = 0
                r.stdout = ""
                return r
            if args[0] == "rocm-smi":
                r = MagicMock()
                r.returncode = 0
                r.stdout = "Card series: Radeon RX 7900 XTX"
                return r
            raise FileNotFoundError

        with patch("subprocess.run", side_effect=fake_run):
            assert detect_gpu_vendor() == "nvidia"


# ---------------------------------------------------------------------------
# TestRocmLibsAvailable
# ---------------------------------------------------------------------------

class TestRocmLibsAvailable:
    def test_all_libs_present(self):
        with patch("ctypes.CDLL", return_value=MagicMock()):
            ok, err = _check_rocm_libs_available()
        assert ok is True
        assert err is None

    def test_missing_lib(self):
        def fake_cdll(name):
            if name == "librocblas.so":
                raise OSError("not found")
            return MagicMock()

        with patch("ctypes.CDLL", side_effect=fake_cdll):
            ok, err = _check_rocm_libs_available()
        assert ok is False
        assert "librocblas.so" in err
        assert "ROCm" in err


# ---------------------------------------------------------------------------
# TestRocmCtranslate2
# ---------------------------------------------------------------------------

class TestRocmCtranslate2:
    def test_rocm_ct2_available(self):
        fake_ct2 = MagicMock()
        fake_ct2.get_supported_compute_types.return_value = ["float16", "int8_float16"]
        with patch.dict("sys.modules", {"ctranslate2": fake_ct2}):
            assert _check_rocm_ctranslate2() is True

    def test_standard_ct2(self):
        fake_ct2 = MagicMock()
        fake_ct2.get_supported_compute_types.side_effect = Exception("no ROCm")
        with patch.dict("sys.modules", {"ctranslate2": fake_ct2}):
            assert _check_rocm_ctranslate2() is False


# ---------------------------------------------------------------------------
# TestIsCudaAvailableAmd
# ---------------------------------------------------------------------------

class TestIsCudaAvailableAmd:
    def setup_method(self):
        _reset_caches()

    def _make_ct2(self, compute_types=("float16",)):
        fake = MagicMock()
        fake.get_supported_compute_types.return_value = list(compute_types)
        return fake

    def test_amd_with_rocm_and_ct2(self):
        """Full AMD setup → True."""
        fake_ct2 = self._make_ct2(["float16"])
        with patch.dict("sys.modules", {"ctranslate2": fake_ct2}), \
             patch.object(gpu_module, "detect_gpu_vendor", return_value="amd"), \
             patch.object(gpu_module, "_check_rocm_libs_available", return_value=(True, None)), \
             patch.object(gpu_module, "_check_rocm_ctranslate2", return_value=True):
            assert is_cuda_available() is True

    def test_amd_without_rocm(self):
        """AMD GPU but ROCm libs missing → False."""
        fake_ct2 = self._make_ct2(["float16"])
        with patch.dict("sys.modules", {"ctranslate2": fake_ct2}), \
             patch.object(gpu_module, "detect_gpu_vendor", return_value="amd"), \
             patch.object(gpu_module, "_check_rocm_libs_available",
                          return_value=(False, "ROCm library libamdhip64.so not found. Install ROCm toolkit.")):
            assert is_cuda_available() is False

    def test_amd_with_rocm_no_ct2(self):
        """ROCm libs OK but ctranslate2 not ROCm-built → False."""
        fake_ct2 = self._make_ct2(["float16"])
        with patch.dict("sys.modules", {"ctranslate2": fake_ct2}), \
             patch.object(gpu_module, "detect_gpu_vendor", return_value="amd"), \
             patch.object(gpu_module, "_check_rocm_libs_available", return_value=(True, None)), \
             patch.object(gpu_module, "_check_rocm_ctranslate2", return_value=False):
            assert is_cuda_available() is False


# ---------------------------------------------------------------------------
# TestGetGpuNameAmd
# ---------------------------------------------------------------------------

class TestGetGpuNameAmd:
    def test_amd_gpu_name_from_rocm_smi(self):
        """rocm-smi returns a product name line → parse it correctly."""
        def fake_run(args, **kwargs):
            if args[0] == "nvidia-smi":
                raise FileNotFoundError
            if args[0] == "rocm-smi":
                r = MagicMock()
                r.returncode = 0
                r.stdout = "Card series:          Radeon RX 7900 XTX\n"
                return r
            raise FileNotFoundError

        with patch("subprocess.run", side_effect=fake_run):
            name = get_gpu_name()
        assert name == "Radeon RX 7900 XTX"

    def test_fallback_to_none(self):
        """Both nvidia-smi and rocm-smi fail → None."""
        def fake_run(args, **kwargs):
            raise FileNotFoundError

        with patch("subprocess.run", side_effect=fake_run):
            name = get_gpu_name()
        assert name is None


# ---------------------------------------------------------------------------
# TestGetRocmStatus
# ---------------------------------------------------------------------------

class TestGetRocmStatus:
    def test_get_rocm_status_fully_available(self):
        with patch.object(gpu_module, "_check_rocm_libs_available", return_value=(True, None)), \
             patch.object(gpu_module, "_check_rocm_ctranslate2", return_value=True):
            ok, msg = get_rocm_status()
        assert ok is True
        assert msg == "ROCm available"

    def test_get_rocm_status_libs_no_ct2(self):
        with patch.object(gpu_module, "_check_rocm_libs_available", return_value=(True, None)), \
             patch.object(gpu_module, "_check_rocm_ctranslate2", return_value=False):
            ok, msg = get_rocm_status()
        assert ok is False
        assert "ctranslate2" in msg

    def test_get_rocm_status_no_rocm(self):
        error_msg = "ROCm library libamdhip64.so not found. Install ROCm toolkit."
        with patch.object(gpu_module, "_check_rocm_libs_available", return_value=(False, error_msg)):
            ok, msg = get_rocm_status()
        assert ok is False
        assert msg == error_msg


# ---------------------------------------------------------------------------
# TestGetGpuInfoVendorFields
# ---------------------------------------------------------------------------

class _FakeTranscriptionService:
    def get_current_device(self):
        return "cpu"

    def get_current_compute_type(self):
        return "int8"


def _make_controller():
    """Build a minimal AppController-like object with get_gpu_info() wired up."""
    # Import app_controller via sys.path manipulation
    src = Path(__file__).parent.parent
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    # Stub heavy dependencies before import
    for mod in [
        "services.audio", "services.hotkey", "services.clipboard",
        "services.database", "services.settings", "services.cudnn_downloader",
        "sounddevice", "numpy", "keyboard", "pyloid",
    ]:
        sys.modules.setdefault(mod, MagicMock())

    # Stub transcription service
    fake_transcription = MagicMock()
    fake_transcription.get_current_device.return_value = "cpu"
    fake_transcription.get_current_compute_type.return_value = "int8"

    import importlib
    import services.gpu as gpu_svc
    import app_controller as ac_module

    controller = object.__new__(ac_module.AppController)
    controller.transcription_service = fake_transcription
    return controller, ac_module


class TestGetGpuInfoVendorFields:
    def setup_method(self):
        gpu_module._cuda_available_cache = None
        gpu_module._gpu_vendor_cache = "unset"

    def _get_info(self, vendor, cuda_available, rocm_available, rocm_message,
                  cudnn_available=False, cudnn_message=None, gpu_name=None):
        controller, ac_module = _make_controller()
        with patch.object(gpu_module, "detect_gpu_vendor", return_value=vendor), \
             patch.object(gpu_module, "get_rocm_status", return_value=(rocm_available, rocm_message)), \
             patch("services.gpu.is_cuda_available", return_value=cuda_available), \
             patch("services.gpu.get_cudnn_status", return_value=(cudnn_available, cudnn_message)), \
             patch("services.gpu.get_gpu_name", return_value=gpu_name), \
             patch("services.gpu.get_cuda_compute_types", return_value=[]):
            # Call through the actual method but patch the module-level functions
            import app_controller as ac
            with patch.object(ac, "is_cuda_available", return_value=cuda_available), \
                 patch.object(ac, "get_cudnn_status", return_value=(cudnn_available, cudnn_message)), \
                 patch.object(ac, "get_gpu_name", return_value=gpu_name), \
                 patch.object(ac, "get_cuda_compute_types", return_value=[]), \
                 patch.object(ac, "detect_gpu_vendor", return_value=vendor), \
                 patch.object(ac, "get_rocm_status", return_value=(rocm_available, rocm_message)):
                return controller.get_gpu_info()

    def test_get_gpu_info_includes_vendor_fields(self):
        info = self._get_info("nvidia", True, False, None, cudnn_available=True)
        assert "gpuVendor" in info
        assert "rocmAvailable" in info
        assert "rocmMessage" in info

    def test_get_gpu_info_nvidia_scenario(self):
        info = self._get_info("nvidia", True, False, None, cudnn_available=True)
        assert info["gpuVendor"] == "nvidia"
        assert info["rocmAvailable"] is False

    def test_get_gpu_info_amd_scenario(self):
        info = self._get_info("amd", True, True, "ROCm available")
        assert info["gpuVendor"] == "amd"
        assert info["rocmAvailable"] is True

    def test_get_gpu_info_no_gpu(self):
        info = self._get_info(None, False, False, "ROCm not found",
                              cudnn_available=False, cudnn_message="cuDNN not found")
        assert info["gpuVendor"] is None
        assert info["rocmAvailable"] is False
        assert info["cudnnAvailable"] is False


# ---------------------------------------------------------------------------
# TestValidateDeviceSetting
# ---------------------------------------------------------------------------

class TestValidateDeviceSetting:
    def setup_method(self):
        _reset_caches()

    def test_validate_cuda_with_working_amd(self):
        """AMD fully configured, is_cuda_available() returns True → (True, None)."""
        with patch.object(gpu_module, "is_cuda_available", return_value=True):
            valid, err = validate_device_setting("cuda")
        assert valid is True
        assert err is None

    def test_validate_cuda_amd_no_rocm(self):
        """AMD GPU, ROCm libs missing → error about ROCm not installed."""
        with patch.object(gpu_module, "is_cuda_available", return_value=False), \
             patch.object(gpu_module, "detect_gpu_vendor", return_value="amd"), \
             patch.object(gpu_module, "_check_rocm_libs_available",
                          return_value=(False, "ROCm library not found")):
            valid, err = validate_device_setting("cuda")
        assert valid is False
        assert "ROCm is not installed" in err

    def test_validate_cuda_amd_no_ct2(self):
        """AMD GPU, ROCm libs OK, ctranslate2 lacks ROCm support → error about setup-rocm.sh."""
        with patch.object(gpu_module, "is_cuda_available", return_value=False), \
             patch.object(gpu_module, "detect_gpu_vendor", return_value="amd"), \
             patch.object(gpu_module, "_check_rocm_libs_available", return_value=(True, None)), \
             patch.object(gpu_module, "_check_rocm_ctranslate2", return_value=False):
            valid, err = validate_device_setting("cuda")
        assert valid is False
        assert "ctranslate2 lacks ROCm" in err

    def test_validate_cuda_nvidia_no_cudnn(self):
        """NVIDIA GPU, CUDA detected but cuDNN missing → existing NVIDIA error preserved."""
        fake_ct2 = MagicMock()
        fake_ct2.get_supported_compute_types.return_value = ["float32"]
        with patch.object(gpu_module, "is_cuda_available", return_value=False), \
             patch.object(gpu_module, "detect_gpu_vendor", return_value="nvidia"), \
             patch.dict("sys.modules", {"ctranslate2": fake_ct2}):
            valid, err = validate_device_setting("cuda")
        assert valid is False
        assert "cuDNN is not installed" in err

    def test_validate_cuda_no_gpu(self):
        """No GPU detected → 'No compatible GPU detected.'"""
        with patch.object(gpu_module, "is_cuda_available", return_value=False), \
             patch.object(gpu_module, "detect_gpu_vendor", return_value=None):
            valid, err = validate_device_setting("cuda")
        assert valid is False
        assert "No compatible GPU detected" in err

    def test_validate_cpu_always_valid(self):
        """device='cpu' is always valid regardless of GPU state."""
        with patch.object(gpu_module, "is_cuda_available", return_value=False):
            valid, err = validate_device_setting("cpu")
        assert valid is True
        assert err is None

    def test_validate_auto_always_valid(self):
        """device='auto' is always valid regardless of GPU state."""
        with patch.object(gpu_module, "is_cuda_available", return_value=False):
            valid, err = validate_device_setting("auto")
        assert valid is True
        assert err is None
