# Hydrostatic plate lab
**[Open the interactive app](https://biswashk920.github.io/hydrostatic-plate-lab/)**

Pressure, total force and center of pressure on a flat plate in a fluid.
Python package plus an interactive page in `docs/` (works on GitHub Pages).

**Inputs:** shape (rectangle, circle, right triangle), depth of the top edge
(negative = above the water), tilt 0-180 deg, spin in the plane, fluid density, gravity.
**Outputs:** total force and its horizontal and vertical parts, maximum pressure,
centroid depth, center of pressure (depth, distance below the centroid, sideways offset).

## How it works
Pressure is rho * g * depth, and depth only changes along the slope of the plate.
The plate is cut into thin strips across the slope; summing the strips gives the
force and where it acts, for any convex shape, including plates partly out of the water.
Results are gauge pressure, for one face of the plate.

## Run it
```
pip install -r requirements.txt
python -m pytest
python -m http.server 8000 --directory docs
```
Then open http://localhost:8000.

```python
from hydrostatic_plate import analyze, rectangle
r = analyze(rectangle(2, 1), top_depth=1, tilt_deg=30)
print(r.force, r.cp_depth)
```
