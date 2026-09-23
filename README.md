# Multi-Agent Trading Desk

A research harness where LLM agents design, train and test trading signals under
walk-forward optimization, and where every idea gets checked for overfitting
before anyone believes it.

**Start here: [docs/research-findings.md](docs/research-findings.md).** It is a
dated log of every strategy tested, including the ones that failed and why. The
most recent entry works through why the earnings-call agents missed Tesla:
eight mechanical causes, and a correction to a figure published earlier in the
same log.

## What's in the repo

- **Model-designer loop** ([src/agents/model_designer.py](src/agents/model_designer.py),
  [src/orchestrator/run_loop.py](src/orchestrator/run_loop.py)): an LLM writes a
  model script, which is trained and scored across walk-forward windows
  ([src/wfo/](src/wfo/)). The per-window results go back to the LLM for the next
  iteration.
- **Earnings-call analysis** ([src/agents/earnings_call_analyst.py](src/agents/earnings_call_analyst.py)):
  five specialist agents (sentiment, guidance credibility, competitive position,
  financial health, optionality) read each transcript independently. A
  synthesis step then reports where they agree and where they conflict.
  Trajectory and stance tools sit alongside it.
- **Robustness gate** ([src/research/correlate_earnings_scores.py](src/research/correlate_earnings_scores.py)):
  each score/horizon correlation is reported with Pearson, Spearman and
  drop-top-N sensitivity. A result is flagged when a few outliers drive it.
- **Data pipeline** ([src/data_pipeline/](src/data_pipeline/)): scripts with no
  LLM calls that fetch and prepare market data, earnings calendars, Form 4
  filings and news.

## LLM backends

The agents can run on either backend, chosen with `--backend`:

- `claude`: the local `claude -p` CLI
- `nemotron`: NVIDIA Nemotron through the NIM API. This needs `NVIDIA_API_KEY`
  (see [.env.example](.env.example)).

The v1 earnings-call results in the findings log came from the four-specialist
pipeline on `claude`. The five-specialist v2 re-run uses `nemotron`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
```

Background on how the design and scope changed over time is in
[docs/project-brief.md](docs/project-brief.md).
