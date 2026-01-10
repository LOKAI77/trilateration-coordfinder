"""
Command-line interface for trilateration coordinate finder.
Handles user interaction, progress display, and result formatting.
"""

import sys
import os
import shutil
import time
from .solver import (
    trilateration_objective,
    multi_stage_optimization,
    alternative_optimization,
    geometric_approach,
    additional_refinement,
    verify_solution,
    refine_position
)

LOGO = r"""
  /$$$$$$                                      /$$  /$$$$$$  /$$                 /$$
 /$$__  $$                                    | $$ /$$__  $$|__/                | $$
| $$  \__/  /$$$$$$   /$$$$$$   /$$$$$$   /$$$$$$$| $$  \__/ /$$ /$$$$$$$   /$$$$$$$  /$$$$$$   /$$$$$$
| $$       /$$__  $$ /$$__  $$ /$$__  $$ /$$__  $$| $$$$    | $$| $$__  $$ /$$__  $$ /$$__  $$ /$$__  $$
| $$      | $$  \ $$| $$  \ $$| $$  \__/| $$  | $$| $$_/    | $$| $$  \ $$| $$  | $$| $$$$$$$$| $$  \__/
| $$    $$| $$  | $$| $$  | $$| $$      | $$  | $$| $$      | $$| $$  | $$| $$  | $$| $$_____/| $$
|  $$$$$$/|  $$$$$$/|  $$$$$$/| $$      |  $$$$$$$| $$      | $$| $$  | $$|  $$$$$$$|  $$$$$$$| $$
 \______/  \______/  \______/ |__/       \_______/|__/      |__/|__/  |__/ \_______/ \_______/|__/

                                             by LOKAI77
"""


def is_tty():
    """Check if stdout is a TTY."""
    return sys.stdout.isatty()


def get_terminal_width():
    """Get terminal width, default to 80 if not available."""
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80


def progress_bar_phase1(percent):
    """Display apt-style progress bar for Phase 1."""
    if not is_tty():
        return

    width = get_terminal_width()
    label = f"[Progress {percent:.0f}%] "
    bar_width = width - len(label)
    filled = int(bar_width * percent / 100)
    bar = '#' * filled + '.' * (bar_width - filled)
    sys.stdout.write(f'\r{label}{bar}')
    sys.stdout.flush()


def counter_phase2(iteration, total_checked):
    """Display counter for Phase 2."""
    if not is_tty():
        return

    width = get_terminal_width()
    message = f"Searching coordinates: iteration {iteration}, checked {total_checked}"
    padding = ' ' * (width - len(message))
    sys.stdout.write(f'\r{message}{padding}')
    sys.stdout.flush()


def clear_status():
    """Clear status line."""
    if is_tty():
        width = get_terminal_width()
        sys.stdout.write('\r' + ' ' * width + '\r')
        sys.stdout.flush()


def log(message):
    """Print a log message."""
    clear_status()
    print(message)


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    """Main CLI entry point."""
    clear_screen()
    print(LOGO)
    time.sleep(0.5)

    reference_points = {"lat": [], "lon": []}
    measured_distances = []

    for i in range(3):
        print(f"Reference Point {i+1}:")
        lat = float(input(f"Latitude (degrees): "))
        lon = float(input(f"Longitude (degrees): "))
        distance = float(input(f"Distance to unknown point (meters): "))

        reference_points["lat"].append(lat)
        reference_points["lon"].append(lon)
        measured_distances.append(distance)

    log("Starting Optimization Process (Phase 1)")

    methods = [
        ("Multi-stage optimization", multi_stage_optimization, 25),
        ("Alternative optimization", alternative_optimization, 50),
        ("Geometric approach", geometric_approach, 75)
    ]

    best_solution = None
    best_error = float('inf')
    best_method = None

    for method_name, method_func, progress in methods:
        try:
            progress_bar_phase1(progress - 25)
            log(f"Trying {method_name}...")
            solution = method_func(reference_points, measured_distances)
            error = trilateration_objective(solution, reference_points, measured_distances)

            log(f"Solution: {solution}")
            log(f"Error value: {error}")

            if error < best_error:
                best_error = error
                best_solution = solution
                best_method = method_name
                log("New best solution found")

            progress_bar_phase1(progress)
        except Exception as e:
            log(f"Method {method_name} failed: {str(e)}")

    if best_solution is None:
        clear_status()
        log("All optimization methods failed. Please check your input data.")
        return

    progress_bar_phase1(85)
    log("Performing additional refinement...")
    refined_solution = additional_refinement(best_solution, reference_points, measured_distances)
    refined_error = trilateration_objective(refined_solution, reference_points, measured_distances)

    if refined_error < best_error:
        best_solution = refined_solution
        best_error = refined_error
        best_method += " with refinement"
        log("Refinement improved solution")

    progress_bar_phase1(100)

    estimated_lat, estimated_lon = best_solution
    residuals, rms_error = verify_solution(best_solution, reference_points, measured_distances)

    clear_status()
    log("Trilateration Results (Phase 1)")
    log(f"Best method: {best_method}")
    log(f"Estimated Position: (Lat: {estimated_lat} Lng: {estimated_lon})")
    residual_str = "; ".join([f"{i+1}: {r:.2f}" for i, r in enumerate(residuals)])
    log(f"Residuals (errors in meters): ({residual_str})")
    log(f"RMS Error: {rms_error:.2f}m")

    log("Starting Final Position Refinement (Phase 2)")
    log(f"Starting position: {estimated_lat}, {estimated_lon}")

    for i in range(len(reference_points["lat"])):
        log(f"Reference point {i+1}: {reference_points['lat'][i]}, {reference_points['lon'][i]}")

    for i, d in enumerate(measured_distances):
        log(f"Known distance {i+1}: {d:.2f}m")

    estimated_start = (estimated_lat, estimated_lon)
    final_point, best_errors, iterations, total_checked, best_total_error = refine_position(
        estimated_start,
        reference_points,
        measured_distances,
        progress_callback=counter_phase2
    )

    clear_status()

    log("Final Results")
    log("=" * 60)
    log(f"Converged after {iterations} iterations ({total_checked} coordinates checked)")
    log(f"Total error: {best_total_error:.4f}m")
    log("")
    log(f"Final position: {final_point[0]}, {final_point[1]}")
    log("")

    for i, err in enumerate(best_errors):
        log(f"Reference point {i+1} error: {err:.4f}m")

    log("")
    if any(err > 10000 for err in best_errors):
        log("Status: Failed to calculate distance (failed)")
    elif any(5000 < err <= 10000 for err in best_errors):
        log("Status: Precision refinement failed to pinpoint exact location (non-optimal)")
    elif any(100 < err <= 5000 for err in best_errors):
        log("Status: Precision refinement failed to pinpoint exact location (sub-optimal)")
    elif any(1 < err <= 100 for err in best_errors):
        log("Status: Precision refinement failed to pinpoint exact location (near-optimal)")
    elif all(err < 1 for err in best_errors):
        log("Status: RMS error < 1m, precise result found (optimal)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_status()
        log("\nOperation cancelled by user")
