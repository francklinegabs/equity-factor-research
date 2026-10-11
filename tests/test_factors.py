import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factors import (forward_returns, long_short_weights, momentum_12_1,  # noqa: E402
                     portfolio_returns, quantile_portfolios, zscore)
from metrics import max_drawdown, sharpe, turnover  # noqa: E402


def _monthly(n_months=30, n_stocks=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    rets = rng.normal(0.01, 0.06, size=(n_months, n_stocks))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[f"S{i}" for i in range(n_stocks)])


def test_forward_return_is_next_month():
    px = _monthly()
    fwd = forward_returns(px)
    t = px.index[5]
    assert fwd.loc[t, "S0"] == pytest.approx(px.iloc[6]["S0"] / px.iloc[5]["S0"] - 1)
    assert fwd.iloc[-1].isna().all()


def test_momentum_uses_only_past_prices():
    px = _monthly()
    mom = momentum_12_1(px)
    assert mom.iloc[:12].isna().all().all()
    t = 15
    assert mom.iloc[t]["S3"] == pytest.approx(px.iloc[t - 1]["S3"] / px.iloc[t - 12]["S3"] - 1)
    # Changing the current price must not change today's signal.
    px2 = px.copy()
    px2.iloc[t] *= 3
    assert momentum_12_1(px2).iloc[t].equals(mom.iloc[t])


def test_perfect_signal_ranks_quantiles():
    px = _monthly()
    fwd = forward_returns(px)
    q = quantile_portfolios(fwd, fwd)          # "signal" = the future itself -> must be monotonic
    assert (q.mean().diff().dropna() > 0).all()


def test_long_short_weights_are_dollar_neutral():
    sig = zscore(_monthly().pct_change())
    w = long_short_weights(sig.dropna(how="all"))
    assert np.allclose(w.sum(axis=1), 0)
    assert np.allclose(w.abs().sum(axis=1), 2)


def test_metrics():
    r = pd.Series([0.1, -0.5, 0.2])
    assert max_drawdown(r) == pytest.approx(-0.5)
    assert np.isnan(sharpe(pd.Series([0.01, 0.01, 0.01])))
    w = pd.DataFrame({"A": [1.0, 1.0, -1.0], "B": [-1.0, -1.0, 1.0]})
    to = turnover(w)
    assert to.iloc[0] == pytest.approx(0.0)
    assert to.iloc[1] == pytest.approx(2.0)   # full reversal of a 2x gross book


def test_portfolio_return():
    px = _monthly()
    fwd = forward_returns(px)
    w = pd.DataFrame(0.0, index=fwd.index, columns=fwd.columns)
    w["S0"] = 1.0
    pr = portfolio_returns(w, fwd)
    assert pr.iloc[3] == pytest.approx(fwd.iloc[3]["S0"])
