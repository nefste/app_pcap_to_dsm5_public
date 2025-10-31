"""
Lightweight helper for enriching domains with IAB content taxonomy labels.

This module is not actively imported by the app yet. It provides a ready-to-use
function that calls the Website Classification API once an API key is supplied.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from websiteclassificationapi import WebsiteClassificationAPI
except ImportError:  # pragma: no cover - library is optional
    WebsiteClassificationAPI = None  # type: ignore


API_KEY_PLACEHOLDER = "API_KEY_PLACEHOLDER"


def classify_domain_iab(
    domain: str,
    *,
    api_key: str = API_KEY_PLACEHOLDER,
    timeout: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Look up IAB categories for a domain using WebsiteClassificationAPI.

    Parameters
    ----------
    domain:
        Bare domain or eTLD+1 (e.g. "example.com") to classify.
    api_key:
        API token issued by WebsiteClassificationAPI. Defaults to a placeholder
        string; replace with a real key before invoking this helper.
    timeout:
        Optional request timeout in seconds. Delegated to the API client.

    Returns
    -------
    dict | None
        Dictionary with the API response (IAB categories, confidence, etc.),
        or None when the client is unavailable or the lookup fails.
    """
    if WebsiteClassificationAPI is None:
        return None

    if not api_key or api_key == API_KEY_PLACEHOLDER:
        # Guard against accidental use without a real credential.
        return None

    client = WebsiteClassificationAPI(api_key=api_key, timeout=timeout)
    try:
        return client.classify(domain)
    except Exception:
        return None
