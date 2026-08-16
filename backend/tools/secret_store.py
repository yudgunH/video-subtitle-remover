"""Store runtime API secrets inside the selected portable data directory.

Windows builds protect the value with the current user's DPAPI key. The
encrypted blob stays in the selected folder instead of the OS credential vault.
"""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import os

from backend.tools.app_paths import get_data_path


_memory_secret = ""
_SECRET_FILE = "9router-api-key.dpapi"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob_from_bytes(value: bytes):
    buffer = ctypes.create_string_buffer(value)
    blob = _DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    return blob, buffer


def _crypt_protect(value: bytes) -> bytes:
    source, source_buffer = _blob_from_bytes(value)
    destination = _DataBlob()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(source), None, None, None, None,
        _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(destination)
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(destination.pbData, destination.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(destination.pbData)
        del source_buffer


def _crypt_unprotect(value: bytes) -> bytes:
    source, source_buffer = _blob_from_bytes(value)
    destination = _DataBlob()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None,
        _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(destination)
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(destination.pbData, destination.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(destination.pbData)
        del source_buffer


def _secret_path():
    return get_data_path("secrets", _SECRET_FILE, create_parent=True)


def get_nine_router_api_key():
    env_key = os.getenv("VSR_9ROUTER_API_KEY", "").strip()
    if env_key:
        return env_key
    if _memory_secret:
        return _memory_secret
    if os.name != "nt":
        return ""
    try:
        encrypted = base64.b64decode(_secret_path().read_bytes(), validate=True)
        return _crypt_unprotect(encrypted).decode("utf-8").strip()
    except (OSError, ValueError, UnicodeError):
        return ""


def set_nine_router_api_key(api_key):
    global _memory_secret
    api_key = (api_key or "").strip()
    _memory_secret = api_key
    path = _secret_path()
    try:
        if not api_key:
            path.unlink(missing_ok=True)
        elif os.name == "nt":
            encrypted = _crypt_protect(api_key.encode("utf-8"))
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(base64.b64encode(encrypted))
            os.replace(temporary, path)
        else:
            # Never create a plaintext fallback.
            return False
        return True
    except OSError:
        return False
