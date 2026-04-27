"""Cogram — intent-aware memory layer for LLM agents.

A fork of Graphiti by Zep AI Research (Apache 2.0), with the cogram intent
layer, Engram cache, Redis active memory, knot synthesis, and MCP server
baked in directly. See NOTICE for attribution requirements.
"""
import os as _os

# Disable upstream graphiti's PostHog telemetry by default. Users opt back in
# via GRAPHITI_TELEMETRY_ENABLED=true. Avoids running cogram usage as upstream
# telemetry data.
_os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')

from .main import Cogram, Graphiti  # Graphiti is the backward-compat alias

__version__ = '0.1.0'
__all__ = ['Cogram', 'Graphiti']
