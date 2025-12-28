"""
Core trilateration solver logic.
Contains all optimization algorithms and distance calculations.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from geopy.distance import geodesic, great_circle
import math
from itertools import product
import warnings

warnings.filterwarnings("ignore")

# Constants
EARTH_RADIUS = 6371000
EARTH_FLATTENING = 1/298.257223563


def vincenty_distance(lat1, lon1, lat2, lon2):
    """Calculate distance using Vincenty's formula via geopy's geodesic implementation."""
    try:
        return geodesic((lat1, lon1), (lat2, lon2), ellipsoid='WGS-84').meters
    except ValueError:
        return great_circle((lat1, lon1), (lat2, lon2)).meters


def trilateration_objective(coordinates, reference_points, measured_distances):
    """Objective function for optimization."""
    lat, lon = coordinates
    errors = []

    for i, (ref_lat, ref_lon, distance) in enumerate(zip(reference_points["lat"],
                                                       reference_points["lon"],
                                                       measured_distances)):
        calculated_distance = vincenty_distance(lat, lon, ref_lat, ref_lon)
        relative_error = (calculated_distance - distance) / distance
        errors.append(relative_error)

    return np.sum(np.array(errors) ** 2)


def grid_search(reference_points, measured_distances, resolution=30):
    """Perform a global grid search to find good initial starting points."""
    lats = np.linspace(-90, 90, resolution)
    lons = np.linspace(-180, 180, resolution)

    best_candidates = []
    best_errors = []

    total_points = resolution * resolution

    point_count = 0
    for lat, lon in product(lats, lons):
        error = trilateration_objective([lat, lon], reference_points, measured_distances)
        best_candidates.append((lat, lon))
        best_errors.append(error)
        point_count += 1

    sorted_indices = np.argsort(best_errors)
    return [best_candidates[i] for i in sorted_indices[:5]]


def multi_stage_optimization(reference_points, measured_distances):
    """Multi-stage optimization using differential evolution and L-BFGS-B."""
    bounds = [(-90, 90), (-180, 180)]
    result_global = differential_evolution(
        trilateration_objective,
        bounds,
        args=(reference_points, measured_distances),
        popsize=30,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=1e-8,
        maxiter=1000
    )

    result_local = minimize(
        trilateration_objective,
        result_global.x,
        args=(reference_points, measured_distances),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': False, 'ftol': 1e-15, 'gtol': 1e-15, 'maxiter': 15000}
    )

    return result_local.x


def alternative_optimization(reference_points, measured_distances):
    """Multi-method optimization approach trying multiple starting points."""
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B']
    best_solution = None
    best_error = float('inf')

    weights = [1/(d**2) for d in measured_distances]
    total_weight = sum(weights)
    initial_lat = sum(lat * w for lat, w in zip(reference_points["lat"], weights)) / total_weight
    initial_lon = sum(lon * w for lon, w in zip(reference_points["lon"], weights)) / total_weight

    for method in methods:
        try:
            bounds = [(-90, 90), (-180, 180)] if method == 'L-BFGS-B' else None
            result = minimize(
                trilateration_objective,
                [initial_lat, initial_lon],
                args=(reference_points, measured_distances),
                method=method,
                bounds=bounds,
                options={'disp': False, 'maxiter': 10000}
            )

            if result.fun < best_error:
                best_error = result.fun
                best_solution = result.x
        except:
            pass

    antipodal_lat = -initial_lat
    antipodal_lon = initial_lon + 180 if initial_lon < 0 else initial_lon - 180

    for method in methods:
        try:
            bounds = [(-90, 90), (-180, 180)] if method == 'L-BFGS-B' else None
            result = minimize(
                trilateration_objective,
                [antipodal_lat, antipodal_lon],
                args=(reference_points, measured_distances),
                method=method,
                bounds=bounds,
                options={'disp': False, 'maxiter': 10000}
            )

            if result.fun < best_error:
                best_error = result.fun
                best_solution = result.x
        except:
            pass

    return best_solution


def geometric_approach(reference_points, measured_distances):
    """Geometrically-focused approach using grid search and local optimization."""
    candidates = grid_search(reference_points, measured_distances, resolution=40)

    best_solution = None
    best_error = float('inf')

    for i, (lat, lon) in enumerate(candidates):
        result = minimize(
            trilateration_objective,
            [lat, lon],
            args=(reference_points, measured_distances),
            method='L-BFGS-B',
            bounds=[(-90, 90), (-180, 180)],
            options={'disp': False, 'ftol': 1e-15, 'gtol': 1e-15, 'maxiter': 15000}
        )

        if result.fun < best_error:
            best_error = result.fun
            best_solution = result.x

    return best_solution


def additional_refinement(best_solution, reference_points, measured_distances):
    """Additional refinement using multiple local optimizations."""
    methods = ['Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B']
    tolerances = [1e-8, 1e-10, 1e-12, 1e-15]

    current_solution = best_solution

    for method in methods:
        for tol in tolerances:
            try:
                bounds = [(-90, 90), (-180, 180)] if method == 'L-BFGS-B' else None
                result = minimize(
                    trilateration_objective,
                    current_solution,
                    args=(reference_points, measured_distances),
                    method=method,
                    bounds=bounds,
                    options={'disp': False, 'ftol': tol, 'gtol': tol, 'maxiter': 20000}
                )
                current_solution = result.x
            except:
                pass

    return current_solution


def verify_solution(solution, reference_points, measured_distances):
    """Verify solution by calculating distances and comparing with measured values."""
    lat, lon = solution
    residuals = []

    for i in range(len(measured_distances)):
        calculated_distance = vincenty_distance(
            lat, lon,
            reference_points["lat"][i], reference_points["lon"][i]
        )
        residual = calculated_distance - measured_distances[i]
        residuals.append(residual)

    rms_error = np.sqrt(np.mean(np.array(residuals) ** 2))
    return residuals, rms_error


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance using Haversine formula."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ, Δλ = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(a))


def total_distance_error(lat, lon, ref_points, measured_distances):
    """Calculate total distance error for a given position."""
    errors = []
    for i in range(len(ref_points["lat"])):
        calc_dist = haversine_distance(lat, lon, ref_points["lat"][i], ref_points["lon"][i])
        errors.append(abs(calc_dist - measured_distances[i]))
    return sum(errors), errors


def generate_surrounding_points(base_point, radii):
    """Generate surrounding points at specified radii and bearings."""
    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    return [(
        geodesic(meters=r).destination(base_point, b).latitude,
        geodesic(meters=r).destination(base_point, b).longitude
    ) for r in radii for b in directions]


def refine_position(base_point, ref_points, measured_distances, max_iter=1000, 
                   tolerance=0.00000000000000001, progress_callback=None):
    """
    Refine position using iterative surrounding point search.
    
    Args:
        base_point: Starting (lat, lon) tuple
        ref_points: Dictionary with 'lat' and 'lon' lists
        measured_distances: List of distances in meters
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
        progress_callback: Optional callback function(iteration, total_checked)
    
    Returns:
        Tuple of (final_point, best_errors)
    """
    current = base_point
    radii = [0.00000000000000001, 0.000000000000001, 0.0000000000001, 0.00000000001, 
             0.000000001, 0.0000001, 0.00001, 0.001, 0.1, 1, 10, 50, 100, 500, 
             1000, 2000, 5000, 7500, 10000]
    best_total_error = float('inf')
    improvements = 0
    best_errors = []

    for iteration in range(max_iter):
        surrounding = generate_surrounding_points(current, radii)
        current_total_error, current_errors = total_distance_error(
            current[0], current[1], ref_points, measured_distances
        )

        if iteration == 0:
            best_total_error = current_total_error
            best_errors = current_errors

        best_point = current
        improved = False

        total_checked = (iteration + 1) * 152
        
        if progress_callback:
            progress_callback(iteration + 1, total_checked)

        for i, pt in enumerate(surrounding):
            total_err, errs = total_distance_error(pt[0], pt[1], ref_points, measured_distances)

            if total_err < best_total_error:
                best_total_error = total_err
                best_errors = errs
                best_point = pt
                improved = True
                improvements += 1

        if improved:
            current = best_point
        else:
            break

        if best_total_error < tolerance:
            break

    return current, best_errors, iteration + 1, (iteration + 1) * 152, best_total_error
