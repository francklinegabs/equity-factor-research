"""Performance and signal-quality metrics for monthly long-short factor portfolios."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def ann_return(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (MONTHS / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(r.dropna().std() * np.sqrt(MONTHS))


def sharpe(r: pd.Series) -> float:
    """Annualised Sharpe of monthly excess returns (long-short portfolios are self-financing)."""
    r = r.dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(MONTHS)) if sd > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r.dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def hit_rate(r: pd.Series) -> float:
    r = r.dropna()
    return float((r > 0).mean())


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Dollars traded per dollar of gross book at each rebalance: sum |w_t - w_{t-1}| / sum |w_{t-1}|.

    0 = no change, 1 = half the book replaced, 2 = the whole book sold and rebought (full reversal).
    """
    w = weights.fillna(0)
    return (w.diff().abs().sum(axis=1) / w.abs().sum(axis=1).shift(1)).iloc[1:]


def information_coefficient(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman rank correlation between signal and next-month return, per month."""
    out = {}
    for dt in signal.index:
        s, f = signal.loc[dt], fwd.loc[dt]
        ok = s.notna() & f.notna()
        if ok.sum() >= 20:
            out[dt] = s[ok].rank().corr(f[ok].rank())
    return pd.Series(out, name="ic")


def summarise(r: pd.Series, ic: pd.Series | None = None, to: pd.Series | None = None) -> dict:
    d = {"ann_return": ann_return(r), "ann_vol": ann_vol(r), "sharpe": sharpe(r),
         "max_drawdown": max_drawdown(r), "hit_rate": hit_rate(r), "months": int(r.notna().sum())}
    if ic is not None and len(ic):
        d.update({"ic_mean": float(ic.mean()), "ic_ir": float(ic.mean() / ic.std() * np.sqrt(MONTHS)),
                  "ic_tstat": float(ic.mean() / ic.std() * np.sqrt(len(ic)))})
    if to is not None and len(to):
        d["avg_turnover"] = float(to.mean())
    return d
