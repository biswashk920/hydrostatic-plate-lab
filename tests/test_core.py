import math

import pytest

from hydrostatic_plate import Circle, analyze, rectangle, right_triangle

RHO_G = 1000 * 9.81


def test_vertical_rectangle_at_surface():
    # F = rho g (L/2) A, center of pressure at 2L/3 below the surface
    r = analyze(rectangle(2, 1), 0, 90)
    assert r.force == pytest.approx(RHO_G * 1 * 2, rel=1e-6)
    assert r.cp_depth == pytest.approx(4 / 3, rel=1e-6)
    assert r.force_horizontal == pytest.approx(r.force)
    assert r.force_vertical == pytest.approx(0, abs=1e-6)


def test_flat_plate_uniform_pressure():
    r = analyze(rectangle(2, 1), 3, 0)
    assert r.force == pytest.approx(RHO_G * 3 * 2, rel=1e-9)
    assert r.cp_depth == pytest.approx(3, rel=1e-9)
    assert r.force_vertical == pytest.approx(r.force)


def test_inclined_rectangle():
    r = analyze(rectangle(2, 1), 1, 30)
    assert r.centroid_depth == pytest.approx(1.5, rel=1e-9)
    assert r.force == pytest.approx(RHO_G * 1.5 * 2, rel=1e-6)
    # y_cp = y_c + I/(A y_c) along the slope, then times sin(tilt)
    y_c = 1.5 / math.sin(math.radians(30))
    y_cp = y_c + (1 * 2**3 / 12) / (2 * y_c)
    assert r.cp_depth == pytest.approx(y_cp * 0.5, rel=1e-6)


def test_vertical_circle():
    r = analyze(Circle(1), 2, 90)
    assert r.force == pytest.approx(RHO_G * 3 * math.pi, rel=1e-4)
    assert r.cp_depth == pytest.approx(3 + 1 / 12, rel=1e-4)


def test_spin_swaps_length_and_width():
    r = analyze(rectangle(2, 1), 0, 90, spin_deg=90)  # width runs downhill
    assert r.force == pytest.approx(RHO_G * 0.5 * 2, rel=1e-6)
    assert r.cp_depth == pytest.approx(2 / 3, rel=1e-6)


def test_partly_out_of_water():
    r = analyze(rectangle(3, 1), -1, 90)   # 1 m above water, 2 m below
    assert r.wetted_area == pytest.approx(2, rel=1e-3)
    assert r.force == pytest.approx(RHO_G * 1 * 2, rel=1e-4)
    assert r.cp_depth == pytest.approx(4 / 3, rel=1e-4)


def test_dry_plate_has_no_force():
    r = analyze(rectangle(2, 1), -5, 90)
    assert r.force == 0 and not r.submerged
    assert r.cp_depth is None


def test_tilt_and_mirror_tilt_give_same_force():
    a = analyze(rectangle(2, 1), 1, 40)
    b = analyze(rectangle(2, 1), 1, 140)
    assert a.force == pytest.approx(b.force)


def test_force_scales_with_density():
    w = analyze(rectangle(2, 1), 1, 60, density=1000)
    m = analyze(rectangle(2, 1), 1, 60, density=13600)
    assert m.force / w.force == pytest.approx(13.6)


def test_symmetric_kite_has_no_sideways_offset():
    r = analyze(rectangle(1, 1), 0, 90, spin_deg=45)
    assert r.cp_sideways == pytest.approx(0, abs=1e-9)


def test_triangle_center_of_pressure_shifts_sideways():
    r = analyze(right_triangle(1, 1), 0, 90)   # exact offset is W/12
    assert abs(r.cp_sideways) == pytest.approx(1 / 12, rel=1e-3)


def test_bad_tilt_rejected():
    with pytest.raises(ValueError):
        analyze(rectangle(1, 1), 0, 200)
