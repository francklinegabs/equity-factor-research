"""Backtest momentum, short-term reversal and low-volatility factors on 97 large US stocks.

Monthly rebalance, quintile long-short portfolios, in-sample/out-of-sample split,
sector-neutral variants, transaction costs. Charts go to figures/, tables to output/.
Run:  python src/analysis.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors import (forward_returns, long_short_weights, low_volatility, momentum_12_1,  # noqa: E402
                     month_end_prices, portfolio_returns, quantile_portfolios, sector_neutral_zscore,
                     short_term_reversal, zscore)
from metrics import information_coefficient, max_drawdown, summarise, turnover  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "output"

START = "2015-01-31"           # first month with 12 months of history for momentum
SPLIT = "2020-12-31"           # in-sample up to here, out-of-sample after
COST_BPS = 10                  # one-way cost per dollar traded

COLORS = {"Momentum 12-1": "#1f5c4b", "Reversal 1M": "#b5541c", "Low volatility": "#3d6fb4", "Composite": "#222222"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def load():
    daily = pd.read_csv(ROOT / "data" / "prices.csv", index_col="date", parse_dates=True).sort_index()
    daily = daily.ffill(limit=5)
    sp = pd.read_csv(ROOT / "data" / "sp500_constituents.csv")
    sectors = sp.set_index(sp["Symbol"].str.replace(".", "-", regex=False))["GICS Sector"]
    sectors = sectors.reindex(daily.columns)
    return daily, sectors


def build_signals(daily, monthly, sectors):
    raw = {"Momentum 12-1": momentum_12_1(monthly),
           "Reversal 1M": short_term_reversal(monthly),
           "Low volatility": low_volatility(daily).reindex(monthly.index)}
    plain = {k: zscore(v) for k, v in raw.items()}
    plain["Composite"] = zscore(sum(plain.values()) / 3)
    neutral = {k: sector_neutral_zscore(v, sectors) for k, v in raw.items()}
    neutral["Composite"] = sector_neutral_zscore(sum(zscore(v) for v in raw.values()) / 3, sectors)
    return plain, neutral


def evaluate(signals, fwd, label):
    rows, series, weights_all = [], {}, {}
    for name, sig in signals.items():
        sig = sig.loc[START:]
        w = long_short_weights(sig)
        gross = portfolio_returns(w, fwd)
        to = turnover(w)
        cost = (w.fillna(0).diff().abs().sum(axis=1) * COST_BPS / 1e4).reindex(gross.index).fillna(0)
        net = gross - cost
        ic = information_coefficient(sig, fwd)
        series[name] = net
        weights_all[name] = w
        for period, sl in [("Full", slice(None)), ("In-sample", slice(None, SPLIT)), ("Out-of-sample", slice(SPLIT, None))]:
            g, n_, i_ = gross.loc[sl].dropna(), net.loc[sl].dropna(), ic.loc[sl]
            s = summarise(n_, i_, to.loc[sl])
            s.update({"version": label, "factor": name, "period": period, "gross_sharpe": summarise(g)["sharpe"]})
            rows.append(s)
    return pd.DataFrame(rows), pd.DataFrame(series), weights_all


# ---------------------------------------------------------------- charts
def chart_cumulative(ls: pd.DataFrame, bench: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for c in ls.columns:
        ax.plot((1 + ls[c].dropna()).cumprod(), color=COLORS[c], lw=2.4 if c == "Composite" else 1.6, label=c)
    ax.axvline(pd.Timestamp(SPLIT), color="#777", lw=0.9, ls="--")
    ax.text(pd.Timestamp(SPLIT), ax.get_ylim()[1] * 0.97, "  out-of-sample →", fontsize=8.5, color="#555", va="top")
    ax.axhline(1, color="#555", lw=0.8)
    ax.set_ylabel("Growth of $1 (long-short, net of costs)")
    ax.set_title(f"Quintile long-short factor portfolios, sector-neutral, {COST_BPS} bps costs")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout(); fig.savefig(FIG / "cumulative_long_short.png"); plt.close(fig)


def chart_quintiles(qret: dict):
    fig, axes = plt.subplots(1, len(qret), figsize=(12, 3.4), sharey=True)
    for ax, (name, q) in zip(axes, qret.items()):
        m = q.mean() * 12 * 100
        ax.bar([f"Q{int(i)}" for i in m.index], m.values, color=[COLORS[name] if i in (1, 5) else "#c9d8d2" for i in m.index])
        ax.set_title(name, fontsize=10)
        ax.axhline(0, color="#555", lw=0.8)
    axes[0].set_ylabel("Avg. annualised return (%)")
    fig.suptitle("Is the signal monotonic? Average return by signal quintile (Q5 = highest signal)")
    fig.tight_layout(); fig.savefig(FIG / "quintile_returns.png"); plt.close(fig)


def chart_ic(signals, fwd):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for name, sig in signals.items():
        ic = information_coefficient(sig.loc[START:], fwd)
        ax.plot(ic.rolling(12).mean(), color=COLORS[name], lw=2.2 if name == "Composite" else 1.4, label=name)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_ylabel("Rolling 12-month mean IC")
    ax.set_title("Signal strength over time (Spearman IC with next-month return)")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout(); fig.savefig(FIG / "rolling_ic.png"); plt.close(fig)


def chart_corr(ls: pd.DataFrame):
    c = ls.dropna().corr()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(c)), c.columns)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(c.iloc[i, j]) > 0.6 else "#222")
    ax.grid(False)
    ax.set_title("Correlation of factor returns")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(FIG / "factor_correlation.png"); plt.close(fig)
    return c


def main() -> None:
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    daily, sectors = load()
    monthly = month_end_prices(daily)
    fwd = forward_returns(monthly)
    print(f"{daily.shape[1]} stocks, {daily.index.min():%Y-%m-%d} to {daily.index.max():%Y-%m-%d}; "
          f"sectors known for {sectors.notna().sum()}; sector counts:\n{sectors.value_counts().to_string()}\n")

    plain, neutral = build_signals(daily, monthly, sectors)
    res_plain, ls_plain, _ = evaluate(plain, fwd, "raw")
    res_neut, ls_neut, w_neut = evaluate(neutral, fwd, "sector-neutral")
    results = pd.concat([res_plain, res_neut], ignore_index=True)
    cols = ["version", "factor", "period", "ann_return", "ann_vol", "sharpe", "gross_sharpe", "max_drawdown",
            "hit_rate", "ic_mean", "ic_ir", "ic_tstat", "avg_turnover", "months"]
    results = results[cols]
    results.round(4).to_csv(OUT / "factor_results.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(results.round(3).to_string(index=False), "\n")

    bench = fwd.loc[START:].mean(axis=1).dropna()
    print("Equal-weight universe benchmark:", {k: round(v, 3) for k, v in summarise(bench).items()})

    qret = {name: quantile_portfolios(sig.loc[START:], fwd) for name, sig in neutral.items()}
    for name, q in qret.items():
        print(f"{name:<15} quintile ann. returns (%):", (q.mean() * 12 * 100).round(2).to_dict())
    pd.concat({k: v.mean() * 12 for k, v in qret.items()}, axis=1).round(4).to_csv(OUT / "quintile_returns.csv")

    chart_cumulative(ls_neut, bench)
    chart_quintiles(qret)
    chart_ic(neutral, fwd)
    c = chart_corr(ls_neut)
    print("\nFactor return correlation\n", c.round(2))

    # Calendar-year returns for the sector-neutral composite and its components
    yearly = ls_neut.groupby(ls_neut.index.year).apply(lambda d: (1 + d).prod() - 1)
    yearly.round(4).to_csv(OUT / "calendar_year_returns.csv")
    print("\nCalendar-year returns\n", (yearly * 100).round(1))

    # Cost sensitivity for the composite
    w = w_neut["Composite"]
    gross = portfolio_returns(w, fwd)
    traded = w.fillna(0).diff().abs().sum(axis=1).reindex(gross.index).fillna(0)
    sens = {bps: summarise(gross - traded * bps / 1e4)["sharpe"] for bps in [0, 5, 10, 20, 30, 50]}
    pd.Series(sens, name="sharpe").to_csv(OUT / "cost_sensitivity.csv")
    print("\nComposite Sharpe vs one-way cost (bps):", {k: round(v, 2) for k, v in sens.items()})
    print("Composite worst drawdown:", round(max_drawdown(ls_neut['Composite']), 3))


if __name__ == "__main__":
    main()
