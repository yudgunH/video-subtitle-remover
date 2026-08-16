import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.tools import app_paths, secret_store


@unittest.skipUnless(os.name == "nt", "Windows DPAPI test")
class SecretStoreTest(unittest.TestCase):
    def test_api_key_round_trip_uses_selected_data_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory) / "data"
            with patch.dict(
                os.environ,
                {app_paths.DATA_DIR_ENV: str(data_root), "VSR_9ROUTER_API_KEY": ""},
                clear=False,
            ):
                secret_store._memory_secret = ""
                self.assertTrue(secret_store.set_nine_router_api_key("test-key-123"))
                secret_store._memory_secret = ""
                self.assertEqual(secret_store.get_nine_router_api_key(), "test-key-123")
                secret_path = data_root / "secrets" / "9router-api-key.dpapi"
                self.assertTrue(secret_path.is_file())
                self.assertNotIn(b"test-key-123", secret_path.read_bytes())
                self.assertTrue(secret_store.set_nine_router_api_key(""))
                self.assertFalse(secret_path.exists())


if __name__ == "__main__":
    unittest.main()
