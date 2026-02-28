"""
proxy/app/analysis — Async deep analysis (Phase 4).

Components here run in background threads AFTER the current LLM call has been
forwarded. They NEVER block the hot path.

The session-flag pattern
------------------------
classify() runs via asyncio.create_task() (non-blocking).
If injection is detected it sets session.ml_injection_flag = True.
intercept.py checks that flag at the START of the NEXT LLM call and BLOCKs.

This protects against multi-turn attack chains while adding zero latency to
the current call.  Single-turn direct attacks are handled by the sync
enforcement layer + canary + output_scanner.

Install
-------
    pip install 'agentblackbox-proxy[classifier]'
    # requires: transformers, torch
"""
from __future__ import annotations

from .classifier import ClassifierResult, classify

__all__ = ["ClassifierResult", "classify"]
