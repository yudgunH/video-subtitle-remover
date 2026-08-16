"""Small credential helper for runtime API secrets.

Secrets are never written to the normal qconfig JSON.  When the optional
system keyring is available they are stored in the OS credential vault;
otherwise they remain in memory for the current app session.
"""

import os


_SERVICE = "video-subtitle-remover"
_NINE_ROUTER_ACCOUNT = "9router-api-key"
_memory_secrets = {}


def get_nine_router_api_key():
    env_key = os.getenv("VSR_9ROUTER_API_KEY", "").strip()
    if env_key:
        return env_key
    if _NINE_ROUTER_ACCOUNT in _memory_secrets:
        return _memory_secrets[_NINE_ROUTER_ACCOUNT]
    try:
        import keyring

        return (keyring.get_password(_SERVICE, _NINE_ROUTER_ACCOUNT) or "").strip()
    except Exception:
        return ""


def set_nine_router_api_key(api_key):
    api_key = (api_key or "").strip()
    if api_key:
        _memory_secrets[_NINE_ROUTER_ACCOUNT] = api_key
    else:
        _memory_secrets.pop(_NINE_ROUTER_ACCOUNT, None)
    try:
        import keyring

        if api_key:
            keyring.set_password(_SERVICE, _NINE_ROUTER_ACCOUNT, api_key)
        else:
            keyring.delete_password(_SERVICE, _NINE_ROUTER_ACCOUNT)
        return True
    except Exception:
        # Memory-only fallback is intentional on headless systems without a
        # keyring backend. The key is still passed directly to the worker.
        return False
