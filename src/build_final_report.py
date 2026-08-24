import json
from pathlib import Path

from src.decision_engine import generate_decision


BASE = Path("/tmp")


def load(name):
    path = BASE / name

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# LOAD PIPELINE RESULTS
# ============================================================

core = load("traffic_report.json")
intel = load("traffic_intelligence.json")
trajectory = load("trajectory_intelligence.json")
flow = load("traffic_flow.json")
speed = load("speed_intelligence.json")


# ============================================================
# EXTRACT CORE DATA
# ============================================================

metrics = core["traffic_metrics"]

traffic = intel["traffic"]
behavior = intel["behavior"]


# ============================================================
# BUILD CANONICAL FINAL REPORT
# ============================================================

final = {
    "system": {
        "name": "Traffic Intelligence System",
        "status": "COMPLETE",
    },

    # --------------------------------------------------------
    # Traffic analytics
    # --------------------------------------------------------

    "traffic": {
        "stable_vehicle_tracks":
            metrics["stable_vehicle_tracks"],

        "peak_active_vehicles":
            metrics["peak_active_stable_vehicles"],

        "average_track_duration_seconds":
            metrics["average_track_duration_seconds"],

        "maximum_track_duration_seconds":
            metrics["maximum_track_duration_seconds"],

        "average_confidence":
            metrics["average_track_confidence"],

        "observed_flow_vehicles_per_minute":
            traffic["observed_flow_vehicles_per_minute"],

        "congestion":
            traffic["congestion"],
    },

    # --------------------------------------------------------
    # Vehicle composition
    # --------------------------------------------------------

    "vehicle_composition":
        core["vehicle_classes"],

    # --------------------------------------------------------
    # Direction analysis
    # --------------------------------------------------------

    "directions":
        core["directions"],

    # --------------------------------------------------------
    # Vehicle behavior
    # --------------------------------------------------------

    "behavior": {
        "stationary_tracks":
            behavior["stationary_tracks"],

        "long_duration_tracks":
            behavior["long_duration_tracks"],

        "potential_stopped_vehicles":
            trajectory["potential_stopped_vehicles"],
    },

    # --------------------------------------------------------
    # Virtual-line traffic flow
    # --------------------------------------------------------

    "flow": {
        "entry_count":
            flow["traffic_flow"]["entry_count"],

        "exit_count":
            flow["traffic_flow"]["exit_count"],

        "flow_vehicles_per_minute":
            flow["traffic_flow"]["flow_vehicles_per_minute"],

        "completed_trips":
            flow["traffic_flow"]["completed_trips"],

        "average_travel_time_seconds":
            flow["traffic_flow"]["average_travel_time_seconds"],
    },

    # --------------------------------------------------------
    # Speed intelligence
    # --------------------------------------------------------

    "speed": {
        "completed_crossings":
            speed["completed_crossings"],

        "average_speed_kmh":
            speed["average_speed_kmh"],

        "minimum_speed_kmh":
            speed["minimum_speed_kmh"],

        "maximum_speed_kmh":
            speed["maximum_speed_kmh"],

        "speed_limit_kmh":
            speed["speed_limit_kmh"],

        "speeding_vehicles":
            speed["speeding_vehicles"],
    },

    # --------------------------------------------------------
    # Traffic events
    # --------------------------------------------------------

    "events":
        intel["events"],
}


# ============================================================
# DECISION INTELLIGENCE
# ============================================================

congestion = traffic["congestion"]
flow_rate = traffic["observed_flow_vehicles_per_minute"]
stationary = behavior["stationary_tracks"]
speeding = speed["speeding_vehicles"]


decision = generate_decision(
    congestion=congestion,
    flow_rate=flow_rate,
    stationary=stationary,
    speeding=speeding,
)


final["decision_intelligence"] = decision


# ============================================================
# SAVE FINAL REPORT
# ============================================================

out = BASE / "traffic_final_report.json"

with open(out, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("=" * 60)
print("FINAL TRAFFIC INTELLIGENCE REPORT")
print("=" * 60)

print(
    f"Stable vehicles : "
    f"{metrics['stable_vehicle_tracks']}"
)

print(
    f"Peak active     : "
    f"{metrics['peak_active_stable_vehicles']}"
)

print(
    f"Flow            : "
    f"{flow_rate:.1f} vehicles/min"
)

print(
    f"Congestion      : "
    f"{congestion}"
)

print(
    f"Average speed   : "
    f"{speed['average_speed_kmh']:.2f} km/h"
)

print(
    f"Speeding        : "
    f"{speeding}"
)

print(
    f"Stationary      : "
    f"{stationary}"
)

print(
    f"System status   : "
    f"{decision['system_status']}"
)


# ============================================================
# RECOMMENDED ACTIONS
# ============================================================

print("\nACTIONS")
print("-" * 60)

for i, action in enumerate(
    decision["actions"],
    1,
):
    print(
        f"{i}. "
        f"[{action['severity']}] "
        f"{action['condition']} -> "
        f"{action['action']}"
    )


print("\nSaved:")
print(out)