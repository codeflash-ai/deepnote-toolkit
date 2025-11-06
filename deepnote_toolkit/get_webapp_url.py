from typing import Optional, Dict

from . import env as dnenv
from .config import get_config

# Cache the config object to avoid repeated I/O/deserialization per process.
_CONFIG_CACHE: Optional[object] = None


def get_project_auth_headers() -> Dict[str, str]:
    """
    Get project authentication headers for detached mode.

    Returns:
        Dict containing RuntimeUuid and Authorization headers
        if in detached mode and environment variables are set, otherwise empty dict.
    """
    headers: Dict[str, str] = {}
    cfg = _get_cached_config()
    runtime = cfg.runtime
    if not runtime.running_in_detached_mode:
        return headers
    project_uuid = runtime.project_id or dnenv.get_env("DEEPNOTE_PROJECT_ID")
    # Only call project_secret getter if detached mode (which we just checked)
    project_secret = runtime.project_secret.get_secret_value()

    if project_uuid:
        headers["RuntimeUuid"] = project_uuid
    if project_secret:
        headers["Authorization"] = f"Bearer {project_secret}"
    return headers


def get_absolute_userpod_api_url(relative_url: str) -> str:
    """Get absolute URL for userpod API endpoint.

    Args:
        relative_url: Relative URL path to append.

    Returns:
        Absolute URL for the userpod API endpoint.
    """
    cfg = _get_cached_config()
    runtime = cfg.runtime
    # Direct mode short-circuit (both detached and dev)
    is_direct_mode = bool(runtime.running_in_detached_mode or runtime.dev_mode)
    if not is_direct_mode:
        return f"http://localhost:19456/userpod-api/{relative_url}"

    # Direct mode requires webapp URL and project ID
    webapp_url = runtime.webapp_url
    if webapp_url:
        webapp_url = webapp_url.rstrip("/")
    project_id = runtime.project_id or dnenv.get_env("DEEPNOTE_PROJECT_ID")

    if not webapp_url or not project_id:
        raise ValueError(
            "DEEPNOTE_WEBAPP_URL and DEEPNOTE_PROJECT_ID must be set in detached mode"
        )

    return f"{webapp_url}/userpod-api/{project_id}/{relative_url}"


def get_absolute_notebook_functions_api_url(relative_url: str) -> str:
    """Get absolute URL for notebook functions API endpoint.

    Args:
        relative_url: Relative URL path to append.

    Returns:
        Absolute URL for the notebook functions API endpoint.
    """
    cfg = get_config()
    is_direct_mode = bool(cfg.runtime.running_in_detached_mode or cfg.runtime.dev_mode)

    if not is_direct_mode:
        return f"http://localhost:19456/api/notebook-functions/{relative_url}"

    webapp_url = get_config().runtime.webapp_url

    if not webapp_url:
        raise ValueError("DEEPNOTE_WEBAPP_URL must be set in detached or dev mode")

    webapp_url = webapp_url.rstrip("/")

    return f"{webapp_url}/api/notebook-functions/{relative_url}"


def _get_cached_config():
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    cfg = get_config()
    _CONFIG_CACHE = cfg
    return cfg
