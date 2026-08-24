from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    8: "Van",
}


def load_results(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_tracks(data):
    tracks = defaultdict(list)

    for frame in data:
        frame_no = frame.get("frame")

        for record in frame.get("tracks", []):
            item = dict(record)
            item["_frame"] = frame_no
            tracks[item["track_id"]].append(item)

    return tracks


def majority_class(records):
    votes = Counter(
        int(r.get("class_id", -1))
        for r in records
    )

    if not votes:
        return -1, {}

    dominant = votes.most_common(1)[0][0]

    return dominant, dict(votes)


def track_summary(track_id, records, fps):
    dominant_class, votes = majority_class(records)

    frames = [
        r["_frame"]
        for r in records
        if r.get("_frame") is not None
    ]

    confidences = [
        float(r.get("confidence", 0.0))
        for r in records
        if r.get("confidence") is not None
    ]

    if frames:
        first_frame = min(frames)
        last_frame = max(frames)
        span = last_frame - first_frame + 1
    else:
        first_frame = None
        last_frame = None
        span = len(records)

    duration = span / fps if fps else 0.0

    return {
        "track_id": int(track_id),
        "frames": len(records),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "span_frames": span,
        "duration_seconds": round(duration, 2),
        "vehicle_class_id": dominant_class,
        "vehicle_type": CLASS_NAMES.get(
            dominant_class,
            f"Unknown({dominant_class})",
        ),
        "class_votes": votes,
        "average_confidence": round(
            sum(confidences) / len(confidences),
            3,
        ) if confidences else 0.0,
    }


def calculate_peak_active(stable_tracks, fps):
    if not stable_tracks:
        return 0, None

    frame_counts = Counter()

    for track in stable_tracks:
        start = track["first_frame"]
        end = track["last_frame"]

        if start is None or end is None:
            continue

        for frame_no in range(start, end + 1):
            frame_counts[frame_no] += 1

    if not frame_counts:
        return 0, None

    peak_frame, peak_count = max(
        frame_counts.items(),
        key=lambda x: x[1],
    )

    return peak_count, peak_frame


def calculate_direction(records):
    if len(records) < 3:
        return "unknown"

    first = records[0]
    last = records[-1]

    try:
        x1, y1 = first["center"]
        x2, y2 = last["center"]
    except (KeyError, TypeError, ValueError):
        return "unknown"

    dx = x2 - x1
    dy = y2 - y1

    threshold = 20

    if abs(dx) < threshold and abs(dy) < threshold:
        return "stationary"

    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"

    return "down" if dy > 0 else "up"


def main():
    parser = argparse.ArgumentParser(
        description="Generate traffic analytics from existing tracking JSON."
    )

    parser.add_argument(
        "--input",
        default="/tmp/traffic_full_core.json",
    )

    parser.add_argument(
        "--output",
        default="/tmp/traffic_report.json",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--min-frames",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 58)
    print("TRAFFIC CORE ANALYTICS")
    print("=" * 58)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Tracking results not found: {input_path}"
        )

    data = load_results(input_path)
    tracks = build_tracks(data)

    print(f"Input frames: {len(data)}")
    print(f"Total track IDs: {len(tracks)}")

    # ----------------------------------------------------------
    # Stable-track filtering
    # ----------------------------------------------------------

    stable = {
        tid: records
        for tid, records in tracks.items()
        if len(records) >= args.min_frames
    }

    fragmented = {
        tid: records
        for tid, records in tracks.items()
        if len(records) < args.min_frames
    }

    print(
        f"Stable tracks (>={args.min_frames} frames): "
        f"{len(stable)}"
    )

    print(
        f"Fragmented tracks: {len(fragmented)}"
    )

    # ----------------------------------------------------------
    # Track-level analytics
    # ----------------------------------------------------------

    summaries = []

    for track_id, records in stable.items():
        summaries.append(
            track_summary(
                track_id,
                records,
                args.fps,
            )
        )

    summaries.sort(
        key=lambda x: x["track_id"]
    )

    # ----------------------------------------------------------
    # Vehicle classes
    # ----------------------------------------------------------

    class_counts = Counter(
        item["vehicle_type"]
        for item in summaries
    )

    # ----------------------------------------------------------
    # Duration
    # ----------------------------------------------------------

    durations = [
        item["duration_seconds"]
        for item in summaries
    ]

    average_duration = (
        sum(durations) / len(durations)
        if durations
        else 0.0
    )

    max_duration = (
        max(durations)
        if durations
        else 0.0
    )

    # ----------------------------------------------------------
    # Peak active stable vehicles
    # ----------------------------------------------------------

    peak_active, peak_frame = calculate_peak_active(
        summaries,
        args.fps,
    )

    # ----------------------------------------------------------
    # Direction analysis
    # ----------------------------------------------------------

    direction_counts = Counter()

    for track_id, records in stable.items():
        direction = calculate_direction(records)
        direction_counts[direction] += 1

        for summary in summaries:
            if summary["track_id"] == int(track_id):
                summary["direction"] = direction
                break

    # ----------------------------------------------------------
    # Longest tracks
    # ----------------------------------------------------------

    longest_tracks = sorted(
        summaries,
        key=lambda x: x["duration_seconds"],
        reverse=True,
    )[:10]

    # ----------------------------------------------------------
    # Confidence
    # ----------------------------------------------------------

    avg_confidence = (
        sum(
            item["average_confidence"]
            for item in summaries
        )
        / len(summaries)
        if summaries
        else 0.0
    )

    # ----------------------------------------------------------
    # Final report
    # ----------------------------------------------------------

    report = {
        "source": str(input_path),

        "configuration": {
            "fps": args.fps,
            "minimum_stable_frames": args.min_frames,
        },

        "dataset": {
            "frames": len(data),
            "total_track_ids": len(tracks),
            "stable_track_ids": len(stable),
            "fragmented_track_ids": len(fragmented),
        },

        "traffic_metrics": {
            "stable_vehicle_tracks": len(stable),
            "peak_active_stable_vehicles": peak_active,
            "peak_frame": peak_frame,
            "average_track_duration_seconds": round(
                average_duration,
                2,
            ),
            "maximum_track_duration_seconds": round(
                max_duration,
                2,
            ),
            "average_track_confidence": round(
                avg_confidence,
                3,
            ),
        },

        "vehicle_classes": dict(class_counts),

        "directions": dict(direction_counts),

        "longest_tracks": longest_tracks,

        "vehicles": summaries,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=4,
        )

    # ----------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------

    print()
    print("=" * 58)
    print("CORE TRAFFIC SUMMARY")
    print("=" * 58)

    print(
        f"Stable vehicle tracks: "
        f"{len(stable)}"
    )

    print(
        f"Peak active vehicles: "
        f"{peak_active}"
    )

    print(
        f"Average track duration: "
        f"{average_duration:.2f}s"
    )

    print(
        f"Maximum track duration: "
        f"{max_duration:.2f}s"
    )

    print(
        f"Average confidence: "
        f"{avg_confidence:.3f}"
    )

    print()
    print("Vehicle classes:")

    for name, count in class_counts.most_common():
        print(
            f"  {name}: {count}"
        )

    print()
    print("Directions:")

    for direction, count in direction_counts.most_common():
        print(
            f"  {direction}: {count}"
        )

    print()
    print("Longest stable tracks:")

    for item in longest_tracks:
        print(
            f"  ID {item['track_id']}: "
            f"{item['vehicle_type']} | "
            f"{item['duration_seconds']:.1f}s | "
            f"{item['frames']} frames | "
            f"{item['direction']}"
        )

    print()
    print(
        f"Saved core report to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
