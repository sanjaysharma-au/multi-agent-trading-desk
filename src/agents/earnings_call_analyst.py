import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("earnings_call_analysis")
DEFAULT_MODEL = "sonnet"
CLAUDE_TIMEOUT_SECONDS = 600

SENTIMENT_SYSTEM_PROMPT = """You are a sentiment and tone analyst reviewing an earnings call transcript. You do not have access to the stock price, analyst ratings, or any information beyond the transcript text itself. Do not use any knowledge of this stock's price before or after this call, even if you recall it from training data -- deliberately set that aside. Reason only from what's said on the call.

Analyze ONLY:
- Management's confidence level in prepared remarks vs. under analyst questioning -- does tone shift when pressed?
- Hedging language ("we believe," "we hope," "should," "targeting" vs. firm commitments)
- Defensiveness, deflection, or evasiveness in response to specific analyst questions -- quote the exact question and answer where this occurs
- Notable enthusiasm or concern that stands out relative to the rest of the call

Ground every claim in a specific quote from the transcript (speaker name + quoted text). Do not speculate about the stock or make a buy/sell judgment -- your job is characterizing tone and confidence, nothing else.

End with a single-paragraph summary: was management's tone overall confident, hedged, or mixed, and where exactly did it shift.
"""

GUIDANCE_SYSTEM_PROMPT = """You are a guidance-credibility analyst reviewing an earnings call transcript. You do not have access to the stock price or any information beyond the transcript text itself. You may draw on your general knowledge of this company's OPERATIONAL track record -- whether it has previously hit or missed similar guidance, delivery targets, or product timelines -- but never on the stock's price performance or how the market reacted to past calls. Do not use any knowledge of this stock's price before or after this call, even if you recall it from training data -- deliberately set that aside.

Analyze ONLY:
- Every specific forward-looking claim made on the call: a date, a number, a target, a timeline. Quote it exactly.
- For each one, assess plausibility: is this consistent with the pace of execution described elsewhere on the same call? Is it consistent with this company's known track record of hitting or missing similar past guidance?
- Flag any guidance that sounds like a walk-back, quiet revision, or reframing of a previous promise, even if not explicitly labeled as such
- Distinguish between guidance backed by concrete evidence stated on the call (e.g. "trial production has already begun") vs. guidance that is pure aspiration

Ground every claim in a specific quote. End with a single-paragraph summary: which specific pieces of guidance are most credible, which are least credible, and why.
"""

COMPETITIVE_SYSTEM_PROMPT = """You are a competitive-positioning analyst reviewing an earnings call transcript. You do not have access to the stock price or any information beyond the transcript text itself. Do not use any knowledge of this stock's price before or after this call, even if you recall it from training data -- deliberately set that aside. Reason only from what's said on the call.

Analyze ONLY:
- Any claims made about competitive moat, differentiation, or market share
- Any comparison (explicit or implied) to competitors
- Any discussion of pricing power, demand vs. supply constraints, or customer switching costs
- Whether these claims are backed by specific evidence stated on the call (numbers, examples) or are unsupported assertions

Ground every claim in a specific quote. End with a single-paragraph summary: what does this call reveal about the company's actual competitive position, separating well-evidenced claims from unsupported ones.
"""

FINANCIAL_SYSTEM_PROMPT = """You are a financial-health analyst reviewing an earnings call transcript. You do not have access to the stock price or any information beyond the transcript text itself. Do not use any knowledge of this stock's price before or after this call, even if you recall it from training data -- deliberately set that aside. Reason only from what's said on the call.

Analyze ONLY the actual numbers and financial statements discussed on the call:
- Cash position, cash flow (operating and free cash flow), and any trend described
- Margins (gross, operating) and what management attributes changes to
- Capex commitments and how they're being funded
- Any language bearing on solvency, liquidity, or the need for future capital raises
- Whether the numbers stated support the qualitative claims made elsewhere on the call (e.g. does the cash flow story match the confidence in prepared remarks?)

Ground every claim in a specific quote or number from the transcript. End with a single-paragraph summary: is the financial picture presented on this call strong, weak, or mixed, and what's the single most important number from this call.
"""

SYNTHESIS_SYSTEM_PROMPT = """You are the lead analyst synthesizing four independent specialist reviews of the same earnings call transcript: a sentiment/tone read, a guidance-credibility read, a competitive-positioning read, and a financial-health read. Each was written without seeing the others.

Your job is NOT to average these into a single bland score. Your job is to:
1. Identify where the four reads AGREE and reinforce each other -- these are the highest-confidence findings.
2. Identify where they GENUINELY CONFLICT -- e.g. a confident/optimistic tone read paired with a weak guidance-credibility or financial-health read is a real tension worth surfacing explicitly, not smoothing over. Name the specific conflict and both sides of it.
3. Note anything one specialist flagged that the others were silent on, and whether that silence is itself informative.

Produce a final synthesis with these sections:
- Where the reads agree (and why that's meaningful)
- Where the reads conflict (state both sides explicitly -- do not resolve the tension by picking a winner unless the evidence clearly supports one side)
- Overall read: a paragraph that reflects the actual uncertainty/tension found above, not a confident single-number verdict this analysis doesn't support

Do not introduce any information beyond what's in the four specialist reports provided to you -- including any knowledge of this stock's actual price performance before or after this call, even from training data. This synthesis must be judgeable purely on the call's own content.
"""

SPECIALISTS = {
    "sentiment": SENTIMENT_SYSTEM_PROMPT,
    "guidance": GUIDANCE_SYSTEM_PROMPT,
    "competitive": COMPETITIVE_SYSTEM_PROMPT,
    "financial": FINANCIAL_SYSTEM_PROMPT,
}


def _call_claude(system_prompt: str, user_input: str, model: str = DEFAULT_MODEL) -> str:
    result = subprocess.run(
        [
            "claude", "-p", user_input,
            "--system-prompt", system_prompt,
            "--model", model,
            "--output-format", "text",
            "--restricted",
            "--disallowedTools", "Bash", "Edit", "Write", "NotebookEdit",
            "--permission-prompts", "none",
        ],
        capture_output=True,
        text=True,
        timeout=CLAUDE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {result.stderr}")
    return result.stdout.strip()


def run_specialist(name: str, transcript_text: str, model: str = DEFAULT_MODEL) -> str:
    system_prompt = SPECIALISTS[name]
    instructions = f"Here is the earnings call transcript to analyze:\n\n{transcript_text}"
    return _call_claude(system_prompt, instructions, model=model)


def run_all_specialists(transcript_text: str, model: str = DEFAULT_MODEL) -> dict[str, str]:
    with ThreadPoolExecutor(max_workers=len(SPECIALISTS)) as executor:
        futures = {
            name: executor.submit(run_specialist, name, transcript_text, model)
            for name in SPECIALISTS
        }
        return {name: future.result() for name, future in futures.items()}


def synthesize(specialist_outputs: dict[str, str], model: str = DEFAULT_MODEL) -> str:
    sections = "\n\n".join(
        f"=== {name.upper()} ANALYST ===\n{text}" for name, text in specialist_outputs.items()
    )
    instructions = f"Here are the four independent specialist reports:\n\n{sections}"
    return _call_claude(SYNTHESIS_SYSTEM_PROMPT, instructions, model=model)


def analyze_transcript(transcript_path: Path, ticker: str, model: str = DEFAULT_MODEL) -> Path:
    transcript_text = transcript_path.read_text()
    specialist_outputs = run_all_specialists(transcript_text, model=model)
    synthesis = synthesize(specialist_outputs, model=model)

    tag = f"{ticker}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    out_dir = OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in specialist_outputs.items():
        (out_dir / f"{name}.md").write_text(text)
    (out_dir / "synthesis.md").write_text(synthesis)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    out_dir = analyze_transcript(args.transcript, args.ticker, model=args.model)
    print(f"Analysis written to {out_dir}")


if __name__ == "__main__":
    main()
