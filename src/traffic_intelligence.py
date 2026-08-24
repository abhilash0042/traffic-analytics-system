from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def classify_congestion(peak_active: int, average_duration: float):
    """
    Simple explainable congestion heuristic.

    This is deliberately conservative because we do not yet
    have calibrated road capacity / lane geometry.
    """
    if peak_active >= 20 or average_duration >= 8:
        return "HIGH"

    if peak_active >= 10 or average_duration >= 5:
        return "MEDIUM"

    return "LOW"


def calculate_flow(
    stable_vehicle_count: int,
    video_seconds: float,
):
    if video_seconds <= 0:
        return 0.0

    return round(
        stable_vehicle_count
        / video_seconds
        * 60.0,
        1,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate traffic intelligence from traffic_report.json."
    )

    parser.add_argument(
        "--input",
        default="/tmp/traffic_report.json",
    )

    parser.add_argument(
        "--output",
        default="/tmp/traffic_intelligence.json",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Traffic report not found: {input_path}"
        )

    report = load_json(input_path)

    dataset = report.get("dataset", {})
    metrics = report.get("traffic_metrics", {})
    classes = report.get("vehicle_classes", {})
    directions = report.get("directions", {})
    vehicles = report.get("vehicles", [])

    fps = float(
        report.get(
            "configuration",
            {},
        ).get(
            "fps",
            25,
        )
    )

    frames = int(
        dataset.get(
            "frames",
            0,
        )
    )

    stable_count = int(
        metrics.get(
            "stable_vehicle_tracks",
            0,
        )
    )

    peak_active = int(
        metrics.get(
            "peak_active_stable_vehicles",
            0,
        )
    )

    average_duration = float(
        metrics.get(
            "average_track_duration_seconds",
            0.0,
        )
    )

    max_duration = float(
        metrics.get(
            "maximum_track_duration_seconds",
            0.0,
        )
    )

    video_seconds = (
        frames / fps
        if fps > 0
        else 0.0
    )

    flow = calculate_flow(
        stable_count,
        video_seconds,
    )

    congestion = classify_congestion(
        peak_active,
        average_duration,
    )

    # ----------------------------------------------------------
    # Direction percentages
    # ----------------------------------------------------------

    total_directional = sum(
        directions.values()
    )

    direction_percentages = {}

    for direction, count in directions.items():
        direction_percentages[direction] = round(
            count / total_directional * 100,
            1,
        ) if total_directional else 0.0

    # ----------------------------------------------------------
    # Vehicle composition percentages
    # ----------------------------------------------------------

    total_classes = sum(
        classes.values()
    )

    class_percentages = {}

    for vehicle_type, count in classes.items():
        class_percentages[vehicle_type] = round(
            count / total_classes * 100,
            1,
        ) if total_classes else 0.0

    # ----------------------------------------------------------
    # Stopped vehicles
    # ----------------------------------------------------------

    stopped = [
        vehicle
        for vehicle in vehicles
        if vehicle.get("direction") == "stationary"
    ]

    # ----------------------------------------------------------
    # Long-duration vehicles
    # ----------------------------------------------------------

    long_duration = [
        vehicle
        for vehicle in vehicles
        if vehicle.get(
            "duration_seconds",
            0,
        ) >= 5
    ]

    # ----------------------------------------------------------
    # Traffic events
    # ----------------------------------------------------------

    events = []

    if peak_active >= 10:
        events.append(
            {
                "type": "HIGH_TRAFFIC_DENSITY",
                "severity": "HIGH"
                if peak_active >= 20
                else "MEDIUM",
                "value": peak_active,
                "message": (
                    f"Peak active vehicles reached "
                    f"{peak_active}."
                ),
            }
        )

    if congestion != "LOW":
        events.append(
            {
                "type": "CONGESTION",
                "severity": congestion,
                "value": average_duration,
                "message": (
                    f"Average stable-track duration "
                    f"was {average_duration:.2f}s."
                ),
            }
        )

    if stopped:
        events.append(
            {
                "type": "STOPPED_VEHICLES",
                "severity": "MEDIUM",
                "value": len(stopped),
                "message": (
                    f"{len(stopped)} vehicle track(s) "
                    "were classified as stationary."
                ),
            }
        )

    # ----------------------------------------------------------
    # Intelligence report
    # ----------------------------------------------------------

    intelligence = {
        "source": str(input_path),

        "video": {
            "frames": frames,
            "fps": fps,
            "duration_seconds": round(
                video_seconds,
                2,
            ),
        },

        "traffic": {
            "stable_vehicle_tracks": stable_count,
            "peak_active_vehicles": peak_active,
            "average_track_duration_seconds": round(
                average_duration,
                2,
            ),
            "maximum_track_duration_seconds": round(
                max_duration,
                2,
            ),
            "observed_flow_vehicles_per_minute": flow,
            "congestion": congestion,
        },

        "vehicle_composition": {
            "counts": classes,
            "percentages": class_percentages,
        },

        "direction_analysis": {
            "counts": directions,
            "percentages": direction_percentages,
        },

        "behavior": {
            "stationary_tracks": len(stopped),
            "long_duration_tracks": len(long_duration),
        },

        "events": events,

        "top_tracks": report.get(
            "longest_tracks",
            [],
        ),
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            intelligence,
            f,
            indent=4,
        )

    # ----------------------------------------------------------
    # Console output
    # ----------------------------------------------------------

    print()
    print("=" * 58)
    print("TRAFFIC INTELLIGENCE")
    print("=" * 58)

    print(
        f"Video duration: "
        f"{video_seconds:.1f}s"
    )

    print(
        f"Stable vehicles: "
        f"{stable_count}"
    )

    print(
        f"Peak active vehicles: "
        f"{peak_active}"
    )

    print(
        f"Observed flow: "
        f"{flow} vehicles/min"
    )

    print(
        f"Average track duration: "
        f"{average_duration:.2f}s"
    )

    print(
        f"Congestion: "
        f"{congestion}"
    )

    print()
    print("Vehicle composition:")

    for name, count in classes.items():
        percentage = class_percentages[name]

        print(
            f"  {name}: "
            f"{count} ({percentage}%)"
        )

    print()
    print("Direction distribution:")

    for direction, count in directions.items():
        percentage = direction_percentages[direction]

        print(
            f"  {direction}: "
            f"{count} ({percentage}%)"
        )

    print()
    print(
        f"Stationary tracks: "
        f"{len(stopped)}"
    )

    print(
        f"Tracks >= 5 seconds: "
        f"{len(long_duration)}"
    )

    print()
    print(
        f"Traffic events: "
        f"{len(events)}"
    )

    for event in events:
        print(
            f"  {event['type']} "
            f"→ {event['severity']}"
        )

    print()
    print(
        f"Saved intelligence report to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
