"""Detect a Palantir-style fundamentals inflection: the quarter a company first proves its business
model works, and durably keeps proving it.

Stage per quarter, in order:
  0 - no revenue
  1 - revenue, but not GAAP profitable
  2 - GAAP net income positive, but free cash flow is not (yet) positive
  3 - GAAP net income AND free cash flow both positive

Palantir crossed straight from stage 1 to stage 3 in 2023 Q1 and never looked back. The "adjusted
profit" tier some pipelines insert between stages 1 and 2 has no standardized SEC XBRL tag - non-GAAP
figures live only in press-release text, not structured filing data - so it has no place in this
purely numeric classification. `scoring.py`'s Nemotron pipeline can read a specific quarter's press
release for that kind of qualitative color if it matters for a given company.
"""

from __future__ import annotations

import pandas as pd

NO_REVENUE, REVENUE, GAAP_PROFIT, GAAP_PROFIT_AND_FCF = 0, 1, 2, 3
STAGE_NAMES = {
    NO_REVENUE: "no revenue",
    REVENUE: "revenue, not GAAP profitable",
    GAAP_PROFIT: "GAAP profit, FCF not yet positive",
    GAAP_PROFIT_AND_FCF: "GAAP profit + positive free cash flow",
}


def merge_financials(fundamentals: pd.DataFrame, cashflow: pd.DataFrame | None) -> pd.DataFrame:
    """Join revenue/net income (`load_fundamentals`) with free cash flow (`load_cashflow`)."""
    df = fundamentals[["period_end", "revenue", "net_income"]].copy()
    if cashflow is not None and not cashflow.empty:
        df = df.merge(cashflow[["period_end", "free_cash_flow"]], on="period_end", how="left")
    else:
        df["free_cash_flow"] = pd.NA
    return df.sort_values("period_end").reset_index(drop=True)


def classify_stages(df: pd.DataFrame) -> pd.Series:
    def stage(r) -> int:
        if pd.isna(r.revenue) or r.revenue <= 0:
            return NO_REVENUE
        if pd.isna(r.net_income) or r.net_income <= 0:
            return REVENUE
        if pd.isna(r.free_cash_flow) or r.free_cash_flow <= 0:
            return GAAP_PROFIT
        return GAAP_PROFIT_AND_FCF

    return df.apply(stage, axis=1)


def find_crossing(df: pd.DataFrame, min_pre_quarters: int = 4, min_post_quarters: int = 2,
                  max_wobble_quarters: int = 1, min_pre_unprofitable_frac: float = 0.75) -> dict | None:
    """The first quarter a company reaches stage 3 and durably stays there through to the most
    recent available quarter, or None if no such crossing exists yet.

    Requirements for a hit:
    - Of the `min_pre_quarters` quarters immediately before the crossing, at least
      `min_pre_unprofitable_frac` were genuinely pre-profit (stage 0 or 1). This is what separates a
      real inflection (Palantir, Carvana: a long unprofitable stretch right up until the turn, maybe
      with one earlier blip) from a mature, already-profitable company having an isolated bad quarter
      or normal seasonal cash-flow noise (a one-off GAAP loss from a tax-law change years earlier
      should not count as "proof" a company was ever a pre-profit business).
    - At least `min_post_quarters` quarters have passed since the crossing (otherwise it is too soon
      to call it durable) with no relapse below stage 2, and the most recent quarter is still at
      stage 3. Up to `max_wobble_quarters` individual quarters may sit at stage 2 (profitable, FCF
      dipped) without disqualifying the run.
    """
    if df.empty:
        return None
    df = df.copy()
    df["stage"] = classify_stages(df)
    stages = df["stage"].tolist()
    n = len(stages)
    for i in range(n):
        if stages[i] != GAAP_PROFIT_AND_FCF or i < min_pre_quarters:
            continue
        pre_window = stages[i - min_pre_quarters:i]
        if sum(1 for s in pre_window if s <= REVENUE) / min_pre_quarters < min_pre_unprofitable_frac:
            continue
        tail = stages[i:]
        if len(tail) - 1 < min_post_quarters:
            continue
        if any(s < GAAP_PROFIT for s in tail) or stages[-1] != GAAP_PROFIT_AND_FCF:
            continue  # a relapse to unprofitable, or not still there today
        wobbles = sum(1 for s in tail if s == GAAP_PROFIT)
        if wobbles > max_wobble_quarters:
            continue
        return {
            "crossing_period_end": df.iloc[i]["period_end"],
            "quarters_since_crossing": n - 1 - i,
            "quarters_of_prior_history": i,
            "wobble_quarters": wobbles,
            "latest_revenue": df.iloc[-1]["revenue"],
            "latest_net_income": df.iloc[-1]["net_income"],
            "latest_free_cash_flow": df.iloc[-1]["free_cash_flow"],
        }
    return None
