"""Download ~11 years of daily adjusted closes for a fixed universe of large US stocks.

The universe is 97 well-known large-cap S&P 500 names that have traded since 2014,
fixed in UNIVERSE below so results are reproducible. (A fixed list of today's large caps
carries survivorship bias; the README discusses what that does to the results.)

Writes data/prices.csv (date x ticker, adjusted close).
Run:  python src/fetch_data.py
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

UNIVERSE = """
AAPL MSFT AMZN GOOGL META NVDA BRK-B JPM JNJ V PG UNH HD MA XOM CVX ABBV MRK PEP KO
AVGO COST LLY WMT MCD CSCO ACN TMO ABT DHR NEE ORCL TXN PM CRM NKE LIN UNP HON QCOM
LOW AMGN IBM INTU CAT GS SBUX BA MDT AMT BLK DE GILD ADP ISRG LMT SYK CVS AXP
TJX MDLZ C SPGI BKNG PLD ZTS CB MO ADI DUK SO CI REGN BDX CME PNC USB ECL CSX NOC
ITW EOG SHW APD AON CL ICE GD MCO EMR WM TGT NSC SLB HUM ETN F
""".split()

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1={p1}&period2={p2}&interval=1d&events=div,splits"
P1 = 1388534400  # 2014-01-01
P2 = 1759276800  # 2025-10-01


def fetch(ticker: str) -> dict[str, float]:
    req = urllib.request.Request(URL.format(t=ticker, p1=P1, p2=P2), headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                res = json.load(resp)["chart"]["result"][0]
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    stamps = res["timestamp"]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    out = {}
    for ts, px in zip(stamps, adj):
        if px is not None:
            out[time.strftime("%Y-%m-%d", time.gmtime(ts))] = round(px, 6)
    return out


def main() -> None:
    DATA.mkdir(exist_ok=True)
    series: dict[str, dict[str, float]] = {}
    for i, t in enumerate(UNIVERSE, 1):
        try:
            series[t] = fetch(t)
            print(f"[{i:>3}/{len(UNIVERSE)}] {t:<6} {len(series[t])} days")
        except Exception as exc:
            print(f"[{i:>3}/{len(UNIVERSE)}] {t:<6} FAILED ({exc})")
        time.sleep(0.4)

    dates = sorted({d for s in series.values() for d in s})
    tickers = sorted(series)
    with (DATA / "prices.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + tickers)
        for d in dates:
            w.writerow([d] + [series[t].get(d, "") for t in tickers])
    print(f"Wrote prices for {len(tickers)} tickers x {len(dates)} days")


if __name__ == "__main__":
    main()
