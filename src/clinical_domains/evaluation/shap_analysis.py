from __future__ import annotations


def require_shap():
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install the 'shap' optional dependency to run SHAP analysis.") from exc
