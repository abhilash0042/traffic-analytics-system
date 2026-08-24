def decision(congestion, flow_rate, stationary, speeding):
    actions = []

    if congestion == "HIGH":
        actions.append(("HIGH", "congestion"))
    elif congestion == "MEDIUM":
        actions.append(("MEDIUM", "congestion"))
    else:
        actions.append(("LOW", "congestion"))

    if stationary >= 10:
        actions.append(("HIGH", "stationary"))
    elif stationary > 0:
        actions.append(("MEDIUM", "stationary"))

    if speeding > 0:
        actions.append(("HIGH", "speeding"))
    else:
        actions.append(("LOW", "speeding"))

    if flow_rate >= 100:
        actions.append(("HIGH", "flow"))
    elif flow_rate >= 50:
        actions.append(("MEDIUM", "flow"))
    else:
        actions.append(("LOW", "flow"))

    if any(severity == "HIGH" for severity, _ in actions):
        status = "ALERT"
    elif any(severity == "MEDIUM" for severity, _ in actions):
        status = "MONITOR"
    else:
        status = "NORMAL"

    return status, actions


def test_normal_traffic():
    status, actions = decision(
        congestion="NORMAL",
        flow_rate=30,
        stationary=0,
        speeding=0,
    )

    assert status == "NORMAL"


def test_medium_traffic():
    status, actions = decision(
        congestion="MEDIUM",
        flow_rate=81.8,
        stationary=7,
        speeding=0,
    )

    assert status == "MONITOR"


def test_high_congestion():
    status, actions = decision(
        congestion="HIGH",
        flow_rate=150,
        stationary=0,
        speeding=0,
    )

    assert status == "ALERT"


def test_many_stationary_vehicles():
    status, actions = decision(
        congestion="NORMAL",
        flow_rate=30,
        stationary=12,
        speeding=0,
    )

    assert status == "ALERT"


def test_speeding_vehicle():
    status, actions = decision(
        congestion="NORMAL",
        flow_rate=30,
        stationary=0,
        speeding=1,
    )

    assert status == "ALERT"


def test_moderate_flow():
    status, actions = decision(
        congestion="NORMAL",
        flow_rate=81.8,
        stationary=0,
        speeding=0,
    )

    assert status == "MONITOR"


if __name__ == "__main__":
    tests = [
        test_normal_traffic,
        test_medium_traffic,
        test_high_congestion,
        test_many_stationary_vehicles,
        test_speeding_vehicle,
        test_moderate_flow,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nAll decision-engine tests passed.")
