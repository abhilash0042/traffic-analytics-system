def generate_decision(
    congestion,
    flow_rate,
    stationary,
    speeding,
):
    actions = []

    if congestion == "HIGH":
        actions.append({
            "severity": "HIGH",
            "condition": "HIGH congestion",
            "action": (
                "Prioritize traffic signal optimization "
                "and congestion mitigation"
            ),
        })
    elif congestion == "MEDIUM":
        actions.append({
            "severity": "MEDIUM",
            "condition": "MEDIUM congestion",
            "action": (
                "Monitor traffic density and consider "
                "signal optimization"
            ),
        })
    else:
        actions.append({
            "severity": "LOW",
            "condition": "NORMAL congestion",
            "action": "No congestion intervention required",
        })

    if stationary >= 10:
        actions.append({
            "severity": "HIGH",
            "condition": f"{stationary} stationary vehicle tracks detected",
            "action": (
                "Inspect for possible obstruction or "
                "traffic incident"
            ),
        })
    elif stationary > 0:
        actions.append({
            "severity": "MEDIUM",
            "condition": f"{stationary} stationary vehicle tracks detected",
            "action": (
                "Inspect for possible obstruction, parking, "
                "or traffic incident"
            ),
        })

    if speeding > 0:
        actions.append({
            "severity": "HIGH",
            "condition": f"{speeding} speeding vehicles detected",
            "action": "Flag vehicles for speed enforcement",
        })
    else:
        actions.append({
            "severity": "LOW",
            "condition": "No speeding vehicles detected",
            "action": "No speed enforcement action required",
        })

    if flow_rate >= 100:
        actions.append({
            "severity": "HIGH",
            "condition": f"High observed flow ({flow_rate:.1f} vehicles/min)",
            "action": (
                "Monitor capacity and consider traffic diversion"
            ),
        })
    elif flow_rate >= 50:
        actions.append({
            "severity": "MEDIUM",
            "condition": (
                f"Moderate observed flow "
                f"({flow_rate:.1f} vehicles/min)"
            ),
            "action": "Continue monitoring traffic flow",
        })
    else:
        actions.append({
            "severity": "LOW",
            "condition": (
                f"Low observed flow "
                f"({flow_rate:.1f} vehicles/min)"
            ),
            "action": (
                "Traffic flow is currently within normal range"
            ),
        })

    if any(a["severity"] == "HIGH" for a in actions):
        system_status = "ALERT"
    elif any(a["severity"] == "MEDIUM" for a in actions):
        system_status = "MONITOR"
    else:
        system_status = "NORMAL"

    return {
        "system_status": system_status,
        "actions": actions,
    }
