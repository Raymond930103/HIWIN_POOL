"""
Coordinate Fudge (Edge Bias Correction)
--------------------------------------
Lightweight, configurable correction applied to table coordinates
after pixel→cm mapping, before converting to robot mm.

Use cases:
- Edge regions show systematic over/under‑shoot (e.g., long side too large).
- You want a simple, hand‑tunable function without changing core vision.

How it works:
- Define a symmetric scale around the table center. Center stays fixed;
  positions toward edges are scaled by a factor.
- Two modes are provided: 'axis' (separate X/Y polynomials) and 'radial'
  (scale depends on distance from center).

Tune in place by editing the coefficients below and toggling FUDGE_ENABLED.
"""
from __future__ import annotations

from typing import Tuple

from .table import TABLE_W_CM, TABLE_H_CM


# Master toggle
FUDGE_ENABLED: bool = True

# Mode: 'axis' or 'radial'
FUDGE_MODE: str = 'axis'

# Axis mode coefficients (recommended start)
# fx = 1 + kx*u^2 + kx2*u^4 ; fy = 1 + ky*v^2 + ky2*v^4
# where u = (x-xc)/(W/2), v = (y-yc)/(H/2)
KX: float = -0.03   # negative to shrink near left/right edges
KX2: float = -0.0
KY: float = -0.0     # negative to shrink near top/bottom edges
KY2: float = -0.0

# Radial mode coefficients
# s = 1 + k1*r^2 + k2*r^4 ; r^2 = u^2 + v^2
K1: float = 0.0
K2: float = 0.0


def _axis_fudge(x_cm: float, y_cm: float) -> Tuple[float, float]:
    xc, yc = TABLE_W_CM / 2.0, TABLE_H_CM / 2.0
    if TABLE_W_CM <= 0 or TABLE_H_CM <= 0:
        return x_cm, y_cm
    u = (x_cm - xc) / (TABLE_W_CM / 2.0)
    v = (y_cm - yc) / (TABLE_H_CM / 2.0)
    u2, v2 = u * u, v * v
    fx = 1.0 + KX * u2 + KX2 * (u2 * u2)
    fy = 1.0 + KY * v2 + KY2 * (v2 * v2)
    x2 = xc + (x_cm - xc) * fx
    y2 = yc + (y_cm - yc) * fy
    return x2, y2


def _radial_fudge(x_cm: float, y_cm: float) -> Tuple[float, float]:
    xc, yc = TABLE_W_CM / 2.0, TABLE_H_CM / 2.0
    if TABLE_W_CM <= 0 or TABLE_H_CM <= 0:
        return x_cm, y_cm
    u = (x_cm - xc) / (TABLE_W_CM / 2.0)
    v = (y_cm - yc) / (TABLE_H_CM / 2.0)
    r2 = u * u + v * v
    s = 1.0 + K1 * r2 + K2 * (r2 * r2)
    x2 = xc + (x_cm - xc) * s
    y2 = yc + (y_cm - yc) * s
    return x2, y2


def apply_fudge(x_cm: float, y_cm: float) -> Tuple[float, float]:
    """Apply configured correction; returns (x', y') in cm.

    Place this just before converting to robot mm.
    """
    if not FUDGE_ENABLED:
        return x_cm, y_cm
    if FUDGE_MODE == 'radial':
        return _radial_fudge(x_cm, y_cm)
    # default to axis mode
    return _axis_fudge(x_cm, y_cm)

