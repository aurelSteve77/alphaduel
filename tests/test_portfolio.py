"""Tests for the portfolio ledger."""

from __future__ import annotations

import numpy as np
import pytest

from alphaduel.environments.portfolio import Portfolio


def test_buy_and_sell_with_fees():
    book = Portfolio(n_assets=2, cash=10_000.0, fee_rate=0.01)
    prices = np.array([100.0, 50.0])

    trade = book.execute(np.array([10, 0]), prices)
    assert trade.executed[0] == 10
    assert book.cash == pytest.approx(10_000 - 1000 - 10)
    assert book.holdings[0] == 10

    trade = book.execute(np.array([-5, 0]), prices)
    assert trade.executed[0] == -5
    assert book.holdings[0] == 5


def test_buy_clipped_by_cash():
    book = Portfolio(n_assets=1, cash=250.0, fee_rate=0.0)
    trade = book.execute(np.array([10]), np.array([100.0]))
    assert trade.executed[0] == 2
    assert book.cash == pytest.approx(50.0)


def test_sell_clipped_without_short():
    book = Portfolio(n_assets=1, cash=0.0, fee_rate=0.0, allow_short=False)
    book.holdings[0] = 3
    trade = book.execute(np.array([-10]), np.array([100.0]))
    assert trade.executed[0] == -3
    assert book.holdings[0] == 0
