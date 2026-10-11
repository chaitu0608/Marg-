import numpy as np

from marg.routing.epsilon_greedy import EpsilonGreedyRouter
from marg.routing.round_robin import RoundRobinRouter
from marg.routing.static import StaticRouter
from marg.routing.thompson import SlidingWindowThompsonRouter
from marg.sim.constants import CONTEXTS, GATEWAYS
from marg.sim.traffic import generate_traffic
from marg.sim.world import make_world


def test_routers_select_valid_gateway_and_seeded_determinism():
    for make in (lambda: StaticRouter(2), lambda: RoundRobinRouter(),
                 lambda: EpsilonGreedyRouter(seed=2),
                 lambda: SlidingWindowThompsonRouter(seed=2)):
        one, two = make(), make()
        choices_one = [one.select(("bank_0", "upi"), t) for t in range(30)]
        choices_two = [two.select(("bank_0", "upi"), t) for t in range(30)]
        assert set(choices_one) <= set(GATEWAYS)
        assert choices_one == choices_two


def test_thompson_shifts_to_gateway_with_higher_success_rate():
    router = SlidingWindowThompsonRouter(window=100, seed=8)
    ctx = ("bank_0", "upi")
    counts = dict.fromkeys(GATEWAYS, 0)
    rng = np.random.default_rng(77)
    for t in range(4_000):
        gateway = router.select(ctx, t)
        counts[gateway] += 1
        p = 0.96 if gateway == "gw_c" else 0.50
        router.update(ctx, gateway, bool(rng.random() < p), t)
    assert counts["gw_c"] > 2_000


def test_window_forgets_old_outcomes():
    router = SlidingWindowThompsonRouter(window=20, seed=1)
    ctx = ("bank_0", "card")
    for t in range(20):
        router.update(ctx, "gw_a", True, t)
    assert router.posterior_mean(ctx, "gw_a") > 0.9
    for t in range(20, 40):
        router.update(ctx, "gw_a", False, t)
    assert router.posterior_mean(ctx, "gw_a") < 0.1


def test_strategies_share_world_uniform_draws_and_do_not_import_world_module():
    import inspect

    from marg.routing import epsilon_greedy, round_robin, static, thompson

    assert all("marg.sim.world" not in inspect.getsource(module)
               for module in (epsilon_greedy, round_robin, static, thompson))
    world = make_world(5, 400)
    traffic = generate_traffic(world, 15)
    for router in (StaticRouter(5), RoundRobinRouter(), EpsilonGreedyRouter(seed=5),
                   SlidingWindowThompsonRouter(seed=5)):
        for tick, tx in enumerate(traffic):
            gateway = router.select(tx.context, tick)
            gi = GATEWAYS.index(gateway)
            ci = CONTEXTS.index(tx.context)
            outcome = bool(world.uniforms[tick, gi] < world.rates[tick, gi, ci])
            router.update(tx.context, gateway, outcome, tick)


def test_benchmark_oracle_outage_metric_uses_each_transaction_draw():
    from bench.run_routing import run_one

    results, _ = run_one(7, 5_000)
    assert results["oracle"]["outage_success_rate"] > 0.5
