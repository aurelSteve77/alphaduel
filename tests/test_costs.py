from alphaduel.config.schema import CostConfig
from alphaduel.envs.costs import TransactionCostModel


def test_zero_trade_is_free():
    model = TransactionCostModel(CostConfig())
    assert model.cost(0, 100.0) == 0.0


def test_cost_increases_with_size():
    model = TransactionCostModel(CostConfig())
    small = model.cost(10, 100.0)
    large = model.cost(1000, 100.0)
    assert large > small > 0.0


def test_impact_only_with_adv():
    cfg = CostConfig(slippage_impact_coef=0.5)
    model = TransactionCostModel(cfg)
    no_adv = model.cost(100, 100.0, adv=None)
    with_adv = model.cost(100, 100.0, adv=1000.0)
    assert with_adv > no_adv
