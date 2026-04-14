"""Backward-compatible imports for the unified MPC module.

The canonical implementation now lives in ``home_optimizer.mpc`` so there is a
single authoritative solver module. This file remains only to preserve older
imports.
"""

from .mpc import CombinedMPCController, CombinedMPCSolution

__all__ = ["CombinedMPCController", "CombinedMPCSolution"]

