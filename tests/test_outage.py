import inspect

from marg.routing.outage import ObservedOutageDetector


def test_detector_signals_from_observed_drop_and_recovers_from_observed_successes():
    detector = ObservedOutageDetector(baseline_window=20, detection_window=5)
    events = []
    tick = 0
    for _ in range(25):
        signal = detector.observe("bank_0", "gw_a", True, tick)
        if signal:
            events.append(signal.kind)
        tick += 1
    for _ in range(5):
        signal = detector.observe("bank_0", "gw_a", False, tick)
        if signal:
            events.append(signal.kind)
        tick += 1
    assert events == ["detected"]
    assert ("bank_0", "gw_a") in detector.active

    for _ in range(5):
        signal = detector.observe("bank_0", "gw_a", True, tick)
        if signal:
            events.append(signal.kind)
        tick += 1
    assert events == ["detected", "recovered"]
    assert not detector.active


def test_detector_has_no_access_to_world_rates_or_outage_schedule():
    source = inspect.getsource(ObservedOutageDetector)
    assert "marg.sim.world" not in source
    assert ".rates" not in source
    assert ".outages" not in source


def test_seeded_outage_benchmark_counts_false_alarms_and_recovery_delays():
    from bench.run_outage_detector import evaluate_seed

    result = evaluate_seed(seed=7, horizon=5_000)
    assert result["outage_count"] > 0
    assert result["detected_outage_count"] + result["missed_outage_count"] == result["outage_count"]
    assert result["false_alarm_count"] >= 0
    assert all(delay >= 0 for delay in result["detection_delay_ticks"])
    assert all(delay >= 0 for delay in result["recovery_delay_ticks"])
