"""
Traffic Flow Intelligence

Consumes tracking JSON and estimates:
- virtual-line crossings
- entry count
- exit count
- direction
- travel time between lines
- calibrated flow rate

No model inference.
No additional downloads.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_data(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def center_from_track(track):
    center = track.get("center")

    if center and len(center) >= 2:
        return float(center[0]), float(center[1])

    bbox = track.get("bbox")

    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        )

    return None


def collect_trajectories(data):
    trajectories = defaultdict(list)

    for frame_data in data:
        frame = int(frame_data["frame"])

        for track in frame_data.get("tracks", []):
            point = center_from_track(track)

            if point is None:
                continue

            trajectories[
                int(track["track_id"])
            ].append(
                {
                    "frame": frame,
                    "x": point[0],
                    "y": point[1],
                    "class_id": int(
                        track.get("class_id", -1)
                    ),
                }
            )

    return trajectories


def crossed_horizontal_line(
    previous_y,
    current_y,
    line_y,
):
    """
    Detect crossing of a horizontal virtual line.
    """

    if previous_y < line_y <= current_y:
        return "DOWN"

    if previous_y > line_y >= current_y:
        return "UP"

    return None


def analyze_crossings(
    trajectories,
    entry_y,
    exit_y,
    fps,
):
    crossings = []
    vehicle_summaries = []

    for track_id, points in sorted(
        trajectories.items()
    ):

        if len(points) < 2:
            continue

        points = sorted(
            points,
            key=lambda p: p["frame"],
        )

        entry_crossing = None
        exit_crossing = None
        direction = None

        for previous, current in zip(
            points,
            points[1:],
        ):

            entry_direction = (
                crossed_horizontal_line(
                    previous["y"],
                    current["y"],
                    entry_y,
                )
            )

            exit_direction = (
                crossed_horizontal_line(
                    previous["y"],
                    current["y"],
                    exit_y,
                )
            )

            if (
                entry_crossing is None
                and entry_direction
            ):
                entry_crossing = {
                    "frame": current["frame"],
                    "direction": entry_direction,
                }

            if (
                exit_crossing is None
                and exit_direction
            ):
                exit_crossing = {
                    "frame": current["frame"],
                    "direction": exit_direction,
                }

        first_y = points[0]["y"]
        last_y = points[-1]["y"]

        if last_y > first_y:
            direction = "DOWN"
        elif last_y < first_y:
            direction = "UP"

        travel_time = None

        if (
            entry_crossing
            and exit_crossing
            and exit_crossing["frame"]
            >= entry_crossing["frame"]
        ):
            frame_delta = (
                exit_crossing["frame"]
                - entry_crossing["frame"]
            )

            travel_time = (
                frame_delta / fps
            )

        vehicle_summary = {
            "track_id": track_id,
            "direction": direction,
            "first_frame": points[0]["frame"],
            "last_frame": points[-1]["frame"],
            "entry_crossing": entry_crossing,
            "exit_crossing": exit_crossing,
            "travel_time_seconds": (
                round(travel_time, 2)
                if travel_time is not None
                else None
            ),
        }

        vehicle_summaries.append(
            vehicle_summary
        )

        if entry_crossing:
            crossings.append(
                {
                    "type": "ENTRY",
                    "track_id": track_id,
                    "frame": entry_crossing[
                        "frame"
                    ],
                    "direction": entry_crossing[
                        "direction"
                    ],
                }
            )

        if exit_crossing:
            crossings.append(
                {
                    "type": "EXIT",
                    "track_id": track_id,
                    "frame": exit_crossing[
                        "frame"
                    ],
                    "direction": exit_crossing[
                        "direction"
                    ],
                }
            )

    return crossings, vehicle_summaries


def calculate_flow(
    crossings,
    total_frames,
    fps,
):
    if total_frames <= 0 or fps <= 0:
        return 0.0

    duration_seconds = (
        total_frames / fps
    )

    duration_minutes = (
        duration_seconds / 60
    )

    entries = sum(
        1
        for event in crossings
        if event["type"] == "ENTRY"
    )

    if duration_minutes <= 0:
        return 0.0

    return round(
        entries / duration_minutes,
        2,
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Virtual-line traffic flow analysis"
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Tracking JSON",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--entry-y",
        type=float,
        required=True,
        help="Entry virtual line Y coordinate",
    )

    parser.add_argument(
        "--exit-y",
        type=float,
        required=True,
        help="Exit virtual line Y coordinate",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(
            "traffic_flow.json"
        )
    )

    data = load_data(input_path)

    trajectories = collect_trajectories(
        data
    )

    crossings, vehicle_summaries = (
        analyze_crossings(
            trajectories,
            args.entry_y,
            args.exit_y,
            args.fps,
        )
    )

    if data:
        first_frame = int(
            data[0]["frame"]
        )
        last_frame = int(
            data[-1]["frame"]
        )

        total_frames = (
            last_frame
            - first_frame
            + 1
        )
    else:
        total_frames = 0

    entry_count = sum(
        1
        for event in crossings
        if event["type"] == "ENTRY"
    )

    exit_count = sum(
        1
        for event in crossings
        if event["type"] == "EXIT"
    )

    flow = calculate_flow(
        crossings,
        total_frames,
        args.fps,
    )

    completed_trips = [
        vehicle
        for vehicle in vehicle_summaries
        if vehicle["travel_time_seconds"]
        is not None
    ]

    average_travel_time = (
        sum(
            v["travel_time_seconds"]
            for v in completed_trips
        )
        / len(completed_trips)
        if completed_trips
        else None
    )

    result = {
        "configuration": {
            "fps": args.fps,
            "entry_line_y": args.entry_y,
            "exit_line_y": args.exit_y,
        },
        "traffic_flow": {
            "entry_count": entry_count,
            "exit_count": exit_count,
            "flow_vehicles_per_minute": flow,
            "completed_trips": len(
                completed_trips
            ),
            "average_travel_time_seconds": (
                round(
                    average_travel_time,
                    2,
                )
                if average_travel_time
                is not None
                else None
            ),
        },
        "crossings": crossings,
        "vehicles": vehicle_summaries,
        "notes": [
            (
                "Flow is based on virtual-line "
                "crossings."
            ),
            (
                "Lines must be calibrated to "
                "the actual camera view."
            ),
            (
                "Fragmented detector tracks can "
                "prevent valid crossings."
            ),
        ],
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 50)
    print("VIRTUAL-LINE TRAFFIC FLOW")
    print("=" * 50)

    print(
        f"Entry line: Y={args.entry_y}"
    )

    print(
        f"Exit line:  Y={args.exit_y}"
    )

    print(
        f"Entries: {entry_count}"
    )

    print(
        f"Exits: {exit_count}"
    )

    print(
        f"Flow: {flow} vehicles/min"
    )

    print(
        f"Completed trips: "
        f"{len(completed_trips)}"
    )

    if average_travel_time is not None:
        print(
            f"Average travel time: "
            f"{average_travel_time:.2f}s"
        )
    else:
        print(
            "Average travel time: "
            "not enough complete crossings"
        )

    print()
    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
