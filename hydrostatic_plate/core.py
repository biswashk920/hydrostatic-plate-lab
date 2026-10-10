"""Hydrostatic pressure and force on a flat plate (gauge pressure).

Geometry
--------
The plate is flat. Two angles place it in the tank:

* tilt (0-180 deg): angle between the plate and the horizontal. 90 = vertical.
* spin (any angle): turns the shape inside its own plane. For a rectangle,
  spin 0 puts the length along the slope and spin 90 puts the width along it.

``top_depth`` is the depth of the shallowest point of the plate. It may be
negative (the top edge sticks out above the water).

Method
------
Pressure depends only on depth, and depth only changes along the slope of the
plate. The plate is cut into thin strips across the slope. Each strip has one
depth, so its pressure is simple; summing the strips gives force, centroid
and center of pressure for any shape.

The force is the push of the fluid on ONE face of the plate. Its horizontal
component is F*sin(tilt); its vertical component is F*cos(tilt), positive
meaning downward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

G = 9.81  # m/s^2
FLUIDS = {"water": 1000.0, "seawater": 1025.0, "oil": 900.0, "mercury": 13600.0}


def _rotate(points, spin_deg):
    """Rotate shape points (u, v) by the spin angle; returns columns (s, t)."""
    a = math.radians(spin_deg)
    c, s = math.cos(a), math.sin(a)
    pts = np.asarray(points, dtype=float)
    u, v = pts[:, 0], pts[:, 1]
    return np.column_stack((u * c - v * s, u * s + v * c))


@dataclass(frozen=True)
class Polygon:
    """Convex polygon given as (u, v) points in metres."""

    points: tuple

    def s_range(self, spin_deg):
        s = _rotate(self.points, spin_deg)[:, 0]
        return float(s.min()), float(s.max())

    def chords(self, s, spin_deg):
        """Left and right edge of the shape (across the slope) at each s."""
        p0 = _rotate(self.points, spin_deg)
        p1 = np.roll(p0, -1, axis=0)
        s0, t0, s1, t1 = p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = (s[:, None] - s0) / (s1 - s0)
        valid = (s0 != s1) & (frac >= 0) & (frac <= 1)
        t = t0 + frac * (t1 - t0)
        lo = np.where(valid, t, np.inf).min(axis=1)
        hi = np.where(valid, t, -np.inf).max(axis=1)
        bad = ~np.isfinite(lo)
        lo[bad] = 0.0
        hi[bad] = 0.0
        return lo, hi


@dataclass(frozen=True)
class Circle:
    radius: float

    def s_range(self, spin_deg):
        return -self.radius, self.radius

    def chords(self, s, spin_deg):
        h = np.sqrt(np.maximum(self.radius**2 - s**2, 0.0))
        return -h, h


def rectangle(length, width):
    l, w = length / 2, width / 2
    return Polygon(((-l, -w), (l, -w), (l, w), (-l, w)))


def right_triangle(base, height):
    b, h = base, height
    return Polygon(((-b / 3, -h / 3), (2 * b / 3, -h / 3), (-b / 3, 2 * h / 3)))


@dataclass(frozen=True)
class PlateResult:
    area: float                         # m^2
    wetted_area: float                  # m^2
    extent: float                       # size of the plate along the slope, m
    centroid_depth: float               # m (negative = above the water)
    max_pressure: float                 # Pa, at the deepest point
    force: float                        # N, perpendicular to the plate
    force_horizontal: float             # N
    force_vertical: float               # N, positive = downward
    cp_depth: Optional[float]           # m, None when the plate is dry
    cp_below_centroid: Optional[float]  # m, measured along the plate
    cp_sideways: Optional[float]        # m, offset across the slope

    @property
    def submerged(self) -> bool:
        return self.force > 0


def analyze(shape, top_depth, tilt_deg, spin_deg=0.0, density=1000.0,
            g=G, strips=4000) -> PlateResult:
    """Force and center of pressure on a flat plate (see module docstring)."""
    if not 0 <= tilt_deg <= 180:
        raise ValueError("tilt_deg must be between 0 and 180")
    if density <= 0 or g <= 0:
        raise ValueError("density and g must be positive")
    if strips < 1:
        raise ValueError("strips must be at least 1")

    s_lo, s_hi = shape.s_range(spin_deg)
    extent = s_hi - s_lo
    ds = extent / strips
    s = s_lo + (np.arange(strips) + 0.5) * ds
    t_left, t_right = shape.chords(s, spin_deg)
    d_area = (t_right - t_left) * ds
    t_mid = (t_left + t_right) / 2

    sin_tilt = math.sin(math.radians(tilt_deg))
    pos = s - s_lo                      # distance along the plate from the top edge
    depth = top_depth + pos * sin_tilt
    pressure = density * g * np.maximum(depth, 0.0)

    area = d_area.sum()
    pos_c = (pos * d_area).sum() / area
    t_c = (t_mid * d_area).sum() / area
    force = (pressure * d_area).sum()
    deepest = top_depth + extent * sin_tilt

    if force > 0:
        pos_cp = (pos * pressure * d_area).sum() / force
        t_cp = (t_mid * pressure * d_area).sum() / force
        cp_depth = top_depth + pos_cp * sin_tilt
        cp_below, cp_side = pos_cp - pos_c, t_cp - t_c
    else:
        cp_depth = cp_below = cp_side = None

    tilt = math.radians(tilt_deg)
    return PlateResult(
        area=float(area),
        wetted_area=float(d_area[depth > 0].sum()),
        extent=float(extent),
        centroid_depth=float(top_depth + pos_c * sin_tilt),
        max_pressure=float(density * g * max(deepest, 0.0)),
        force=float(force),
        force_horizontal=float(force * math.sin(tilt)),
        force_vertical=float(force * math.cos(tilt)),
        cp_depth=None if cp_depth is None else float(cp_depth),
        cp_below_centroid=None if cp_below is None else float(cp_below),
        cp_sideways=None if cp_side is None else float(cp_side),
    )
