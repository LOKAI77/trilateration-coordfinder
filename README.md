# Trilateration Coordinate Finder

High-precision geospatial trilateration solver using multiple optimization methods to determine unknown coordinates from known reference points and distances.

## Features

- **Multiple Optimization Methods**: Uses differential evolution, L-BFGS-B, Nelder-Mead, Powell, CG, and BFGS algorithms
- **Two-Phase Refinement**: Initial optimization followed by iterative precision refinement
- **High Precision**: Sub-meter accuracy in optimal conditions
- **Vincenty Distance Calculations**: Uses WGS-84 ellipsoid for accurate geodesic distances
- **Professional CLI**: Clean, apt-style progress indicators and status updates
- **TTY-Aware**: Adapts output based on terminal capabilities

## Installation

### From PyPI

```bash
pip install trilateration-coordfinder
```

## Usage

### Command Line Interface

Run the interactive CLI:

```bash
trilateration
```

You'll be prompted to enter:
1. Three reference points (latitude, longitude)
2. Distance from each reference point to the unknown location (in meters)

### As a Python Library

```python
from trilateration import (
    multi_stage_optimization,
    refine_position,
    verify_solution
)

# Define reference points and distances
reference_points = {
    "lat": [50.8561306, 49.8109078, 48.5883175],
    "lon": [14.7763778, 18.6925036, 12.4846475]
}
measured_distances = [91070, 304700, 218100]  # in meters

# Phase 1: Initial optimization
solution = multi_stage_optimization(reference_points, measured_distances)

# Phase 2: Precision refinement
final_point, errors, iterations, total_checked, total_error = refine_position(
    solution,
    reference_points,
    measured_distances
)

print(f"Final position: {final_point[0]:.8f}, {final_point[1]:.8f}")
```

## How It Works

### Phase 1: Global Optimization

The solver tries three different optimization approaches:

1. **Multi-stage optimization**: Combines differential evolution (global search) with L-BFGS-B (local refinement)
2. **Alternative optimization**: Tests multiple methods from different starting points, including antipodal positions
3. **Geometric approach**: Uses grid search to find candidates, then refines with local optimization

The best solution from these methods is selected and further refined.

### Phase 2: Iterative Refinement

Starting from the Phase 1 solution, the solver:
- Generates surrounding points at multiple radii (from nanometers to kilometers)
- Tests 8 directions (N, NE, E, SE, S, SW, W, NW) at each radius
- Iteratively moves toward the point with minimum total distance error
- Converges when no improvement is found or tolerance is met

## Output Interpretation

The solver provides residuals (distance errors) for each reference point:

- **< 1m**: Optimal - High precision result
- **100-5000m**: Sub-optimal - Good approximation
- **5000-10000m**: Non-optimal - Rough estimate
- **> 10000m**: Failed - Poor solution quality

## Requirements

- Python >= 3.7
- numpy >= 1.21.0
- scipy >= 1.7.0
- geopy >= 2.2.0

## Author

LOKAI77

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Uses Vincenty's formula via `geopy` for accurate geodesic calculations
- Optimization algorithms provided by `scipy`
- Inspired by real-world GPS trilateration challenges
