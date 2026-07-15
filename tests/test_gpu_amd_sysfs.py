import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

fake_comfy = types.ModuleType('comfy')
fake_model_management = types.ModuleType('comfy.model_management')
fake_model_management.get_torch_device_name = lambda device: 'cpu'
fake_model_management.get_torch_device = lambda: None
fake_comfy.model_management = fake_model_management
sys.modules.setdefault('comfy', fake_comfy)
sys.modules.setdefault('comfy.model_management', fake_model_management)

fake_torch = types.ModuleType('torch')
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault('torch', fake_torch)

fake_crystools_pkg = types.ModuleType('crystools')
fake_crystools_pkg.__path__ = [str(repo_root)]
sys.modules['crystools'] = fake_crystools_pkg

fake_general_pkg = types.ModuleType('crystools.general')
fake_general_pkg.__path__ = [str(repo_root / 'general')]
sys.modules['crystools.general'] = fake_general_pkg

spec = importlib.util.spec_from_file_location('crystools.general.gpu', repo_root / 'general' / 'gpu.py')
gpu_module = importlib.util.module_from_spec(spec)
sys.modules['crystools.general.gpu'] = gpu_module
assert spec.loader is not None
spec.loader.exec_module(gpu_module)

CGPUInfo = gpu_module.CGPUInfo


class TestAmdSysfsMetrics(unittest.TestCase):
    def test_gpu_busy_and_vram_are_read_from_sysfs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            device_path = root / 'card1' / 'device'
            device_path.mkdir(parents=True, exist_ok=True)
            (device_path / 'gpu_busy_percent').write_text('72\n', encoding='utf-8')
            (device_path / 'mem_info_vram_used').write_text('8589934592\n', encoding='utf-8')
            (device_path / 'mem_info_vram_total').write_text('17179869184\n', encoding='utf-8')

            info = CGPUInfo.__new__(CGPUInfo)
            self.assertEqual(72.0, info._read_amd_gpu_busy_percent(1, sysfs_root=root))
            self.assertEqual(
                {'total': 17179869184, 'used': 8589934592},
                info._read_amd_vram_info(1, sysfs_root=root),
            )

    def test_temperature_is_read_from_hwmon_when_sensors_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            device_path = root / 'card0' / 'device'
            hwmon_path = device_path / 'hwmon' / 'hwmon1'
            hwmon_path.mkdir(parents=True, exist_ok=True)
            (hwmon_path / 'temp1_input').write_text('55000\n', encoding='utf-8')

            info = CGPUInfo.__new__(CGPUInfo)
            with mock.patch('crystools.general.gpu.shutil.which', return_value=None):
                self.assertEqual(55.0, info._read_amd_temperature(0, sysfs_root=root))


if __name__ == '__main__':
    unittest.main()
