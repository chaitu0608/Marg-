import numpy as np

from marg.sim.traffic import generate_traffic
from marg.sim.world import BANKS, CONTEXTS, GATEWAYS, make_world


def test_world_seeded_and_different_seeds_differ():
    a, b, c = make_world(12, 100), make_world(12, 100), make_world(13, 100)
    np.testing.assert_array_equal(a.rates, b.rates)
    assert not np.array_equal(a.rates, c.rates)


def test_rate_bounds_and_outage_recovery():
    world = make_world(3, 15_000)
    assert float(world.rates.max()) <= 0.990001
    assert world.outages
    # Raw drift is clipped to [0.30, 0.99]; outages then apply their separate multiplier.
    masked = world.rates.copy()
    for event in world.outages:
        gi = GATEWAYS.index(event.gateway)
        for ci, context in enumerate(CONTEXTS):
            if context[0] == event.bank:
                masked[event.start:event.end, gi, ci] = 0.5
    assert float(masked.min()) >= 0.30
    assert float(masked.max()) <= 0.990001
    event = world.outages[0]
    context = next(c for c in CONTEXTS if c[0] == event.bank)
    gi = GATEWAYS.index(event.gateway)
    ci = CONTEXTS.index(context)
    assert world.rates[event.start, gi, ci] < 0.30
    if event.end < world.horizon:
        assert world.rates[event.end, gi, ci] > world.rates[event.start, gi, ci]


def test_outcome_monotone_in_uniform_and_rate_is_probability():
    world = make_world(4, 100)
    gateway, context, t = GATEWAYS[0], (BANKS[0], "upi"), 5
    assert world.outcome(gateway, context, t, 0.01)
    assert not world.outcome(gateway, context, t, 0.99)
    p = world.p_true(gateway, context, t)
    draws = np.linspace(0, 1, 20_000, endpoint=False)
    observed = np.mean([world.outcome(gateway, context, t, float(u)) for u in draws])
    assert abs(observed - p) < 1e-4


def test_traffic_has_non_uniform_mix_and_burst_ticks():
    world = make_world(6, 10_000)
    traffic = generate_traffic(world, 16)
    assert len(traffic) == world.horizon
    assert len({tx.t for tx in traffic[500:504]}) == 1
    assert traffic[504].t - traffic[503].t == 4
