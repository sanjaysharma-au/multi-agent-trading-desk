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


def find_failure(df: pd.DataFrame, min_quarters: int = 8, max_revenue_to_burn: float = 0.1) -> dict | None:
    """The Canoo-shaped mirror image of `find_crossing`: a long operating history that has never
    proven the business model works, with essentially nothing to show for the cash it has burned.

    A company whose moonshot is "will this business model work at all" gives a clean signal in
    both directions: Palantir proved it did; Canoo, over years of filings, never has. The core
    measure is `cumulative revenue / cumulative losses` - how much of everything the company has
    ever burned was ever offset by real revenue. This is deliberately not a "is the recent trend
    improving" check: a company running out of money often shows shrinking losses in its last few
    quarters simply because it is slashing spending to survive, which a naive trend read would
    mistake for a turnaround. A near-zero ratio, held over many quarters, is much harder to fake
    that way.

    Returns None if there isn't enough history, the company ever crossed into a sustained profit
    (that is `find_crossing`'s subject, not this one), it never actually lost money cumulatively, it
    has been net profitable on balance recently, or - importantly - it has *no revenue tagged in any
    quarter of its entire history*. That last case is not a real signal: banks, REITs and insurers
    report their top line as interest or premium income under XBRL concepts this module doesn't look
    at (not the generic "Revenues" concept operating companies use), so they show up with zero
    revenue in every quarter regardless of how profitable they actually are. Without at least one
    quarter of confirmed revenue we have no reliable denominator, so the honest answer is "we can't
    tell," not "this looks like a failure." (Real cases like Canoo and Lordstown Motors both had a
    handful of quarters with genuine, if tiny, revenue - this guard does not exclude them.)
    """
    if len(df) < min_quarters:
        return None
    df = df.copy()
    df["stage"] = classify_stages(df)
    if df["stage"].max() >= GAAP_PROFIT_AND_FCF:
        return None  # it crossed at some point; find_crossing's territory, not this one
    if (df["revenue"].fillna(0) > 0).sum() == 0:
        return None  # no revenue ever tagged - can't compute a meaningful ratio, not a real failure
    if df.tail(min_quarters)["net_income"].fillna(0).sum() > 0:
        return None  # net profitable on balance recently, whatever the revenue reading says

    total_revenue = df["revenue"].fillna(0).clip(lower=0).sum()
    total_losses = -df["net_income"].fillna(0).clip(upper=0).sum()  # sum of loss-quarter magnitudes
    if total_losses <= 0:
        return None  # never actually lost money cumulatively
    ratio = total_revenue / total_losses
    if ratio > max_revenue_to_burn:
        return None  # revenue has offset a meaningful share of the losses; not a clear failure

    return {
        "quarters_of_history": len(df),
        "quarters_with_revenue": int((df["revenue"].fillna(0) > 0).sum()),
        "max_stage_ever_reached": int(df["stage"].max()),
        "cumulative_revenue": float(total_revenue),
        "cumulative_losses": float(total_losses),
        "revenue_to_burn_ratio": float(ratio),
        "latest_net_income": df.iloc[-1]["net_income"],
        "latest_free_cash_flow": df.iloc[-1]["free_cash_flow"],
    }


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
    - The company must have actually lost money overall before the crossing (cumulative net income
      across every prior quarter is negative). Some filers - hospitals, REITs, insurers, telecoms -
      tag their top line under an industry-specific XBRL concept this module doesn't read, so their
      revenue looks like "0" or missing for years and then "appears" the quarter they switch to (or
      this module starts recognizing) a standard concept. Without this guard that switch looks
      exactly like a stage-0-to-stage-3 crossing even for a company that was hugely profitable the
      whole time (real case caught by this check: HCA Healthcare, $13.7B of cumulative net income in
      the years this module had wrongly classified as its unprofitable "before").
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
        if df["net_income"].iloc[:i].fillna(0).sum() > 0:
            continue  # net profitable overall before this point - not a real "before"
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
