"""Signal construction and quantile portfolio backtesting.

All signals at month-end t use only prices available at t; they are paired with the return
from t to t+1 (`fwd`). This one-period offset is the whole defence against look-ahead bias,
and `tests/test_factors.py` checks it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def month_end_prices(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.resample("ME").last()


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score each row, clipped at +/-3 to limit outliers."""
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    return z.clip(-3, 3)


def sector_neutral_zscore(df: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Z-score within each sector so the signal does not just bet on one industry."""
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for _, cols in sectors.reindex(df.columns).groupby(sectors.reindex(df.columns)).groups.items():
        cols = list(cols)
        if len(cols) >= 3:
            out[cols] = zscore(df[cols])
    return out


def momentum_12_1(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return from t-12 to t-1 months: skips the latest month, which tends to reverse."""
    return monthly.shift(1) / monthly.shift(12) - 1


def short_term_reversal(monthly: pd.DataFrame) -> pd.DataFrame:
    """Negative of last month's return: last month's losers tend to bounce."""
    return -(monthly / monthly.shift(1) - 1)


def low_volatility(daily: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Negative of trailing 3-month daily return volatility, sampled at month end."""
    vol = daily.pct_change(fill_method=None).rolling(window, min_periods=int(window * 0.8)).std() * np.sqrt(252)
    return -vol.resample("ME").last()


def forward_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return earned over the month *after* each date: aligned with signals at the same index."""
    return monthly.shift(-1) / monthly - 1


def quantile_portfolios(signal: pd.DataFrame, fwd: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """Equal-weighted forward return of each signal quantile (1 = lowest signal, q = highest)."""
    rows = {}
    for dt in signal.index:
        s, f = signal.loc[dt], fwd.loc[dt]
        ok = s.notna() & f.notna()
        if ok.sum() < q * 4:
            continue
        buckets = pd.qcut(s[ok].rank(method="first"), q, labels=range(1, q + 1))
        rows[dt] = f[ok].groupby(buckets, observed=True).mean()
    return pd.DataFrame(rows).T


def long_short_weights(signal: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """+1/n on the top quantile, -1/n on the bottom quantile, 0 elsewhere."""
    w = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for dt in signal.index:
        s = signal.loc[dt].dropna()
        if len(s) < q * 4:
            w.loc[dt] = np.nan
            continue
        b = pd.qcut(s.rank(method="first"), q, labels=False)
        top, bot = b[b == q - 1].index, b[b == 0].index
        w.loc[dt, top] = 1 / len(top)
        w.loc[dt, bot] = -1 / len(bot)
    return w.dropna(how="all")


def portfolio_returns(weights: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    return (weights * fwd.reindex_like(weights)).sum(axis=1, min_count=1)
