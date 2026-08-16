import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.tools import app_paths


class AppPathsTest(unittest.TestCase):
    def test_selected_root_controls_all_runtime_cache_and_temp_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            sandbox = Path(temporary_directory)
            app_root = sandbox / "portable-app"
            data_root = sandbox / "selected-data"
            app_root.mkdir()
            original_tempdir = tempfile.tempdir
            try:
                with patch.dict(
                    os.environ,
                    {
                        app_paths.APP_ROOT_ENV: str(app_root),
                        app_paths.DATA_DIR_ENV: str(data_root),
                    },
                    clear=False,
                ):
                    result = app_paths.initialize_runtime_environment()
                    self.assertEqual(result, data_root.resolve())
                    self.assertEqual(Path(tempfile.gettempdir()), data_root / "temp")
                    self.assertEqual(Path(os.environ["TORCH_HOME"]), data_root / "cache" / "torch")
                    self.assertEqual(Path(os.environ["PADDLE_HOME"]), data_root / "cache" / "paddle")
                    self.assertEqual(Path(os.environ["HF_HOME"]), data_root / "cache" / "huggingface")
            finally:
                tempfile.tempdir = original_tempdir

    def test_changing_root_migrates_durable_data_and_writes_locator(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            sandbox = Path(temporary_directory)
            app_root = sandbox / "portable-app"
            old_root = sandbox / "old-data"
            new_root = sandbox / "new-data"
            app_root.mkdir()
            (old_root / "config").mkdir(parents=True)
            (old_root / "config" / "config.json").write_text(
                '{"portable": true}', encoding="utf-8"
            )
            original_tempdir = tempfile.tempdir
            try:
                with patch.dict(
                    os.environ,
                    {
                        app_paths.APP_ROOT_ENV: str(app_root),
                        app_paths.DATA_DIR_ENV: str(old_root),
                    },
                    clear=False,
                ):
                    selected = app_paths.set_data_root(new_root)
                    self.assertEqual(selected, new_root.resolve())
                    self.assertTrue((new_root / "config" / "config.json").is_file())
                    locator = json.loads(
                        (app_root / ".vsr-data-location.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(Path(locator["data_root"]), new_root.resolve())
            finally:
                tempfile.tempdir = original_tempdir


if __name__ == "__main__":
    unittest.main()
