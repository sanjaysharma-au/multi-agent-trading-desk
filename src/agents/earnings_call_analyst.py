import argparse
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path("earnings_call_analysis")
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_BACKEND = "nemotron"  # "nemotron" or "claude"
CLAUDE_TIMEOUT_SECONDS = 600
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 10

NIM_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_TIMEOUT_SECONDS = 600
# With stream:False, requests' timeout only fires on total silence from the start
# of the request -- a connection that receives some bytes (e.g. headers) and then
# stalls never trips it (observed: a connection sat ESTABLISHED for 20+ minutes
# after receiving a partial response). Streaming sidesteps this: each SSE chunk is
# its own low-level read, so a plain per-read timeout correctly bounds the GAP
# between chunks instead of the whole response.
NIM_STREAM_STALL_SECONDS = 90
NIM_MAX_TOKENS = 8192
# Reasoning tokens count against the same max_tokens budget as the visible
# content. A prompt that needs a lot of reasoning (many claims to check, say)
# can exhaust the budget before the model ever reaches its answer, leaving a
# non-empty but truncated completion (finish_reason "length") that the old
# empty-completion check didn't catch -- observed: ~37% of trajectory-tool
# calls silently truncated mid-analysis, no verdict, marked as succeeded.
# On a length truncation we retry with a larger budget rather than the same
# one that already proved insufficient for this specific prompt.
NIM_MAX_TOKENS_CEILING = 32768
NIM_TEMPERATURE = 0.2
NIM_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
# integrate.api.nvidia.com throttles this account under concurrent load (observed:
# 5 simultaneous requests -> 503 "Service temporarily overloaded"), so nemotron
# calls get their own lower concurrency cap and more retry headroom than claude.
NIM_MAX_CONCURRENCY = 2
NIM_MAX_ATTEMPTS = 5
NIM_RETRY_BACKOFF_SECONDS = 15

SENTIMENT_SYSTEM_PROMPT = """You are a sentiment and tone analyst reviewing an earnings call transcript. You do not have access to the stock price, analyst ratings, or any information beyond the transcript text itself. Do not use any knowledge of this stock's price before or after this call, even if you recall it from training data -- deliberately set that aside. Reason only from what's said on the call.

Analyze ONLY:
- Management's confidence level in prepared remarks vs. under analyst questioning -- does tone shift when pressed?
- Hedging language ("we believe," "we hope," "should," "targeting" vs. firm commitments)
- Defensiveness, deflection, or evasiveness in response to specific analyst questions -- quote the exact question and answer where this occurs
- Notable enthusiasm or concern that stands out relative to the rest of the call

Ground every claim in a specific quote from the transcript (speaker name + quoted text). Do not speculate about the stock or make a buy/sell judgment -- your job is characterizing tone and confidence, nothing else.

End with a single-paragraph summary: was management's tone overall confident, hedged, or mixed, and where exactly did it shift.
"""

GUIDANCE_SYSTEM_PROMPT = """You are a guidance-credibility analyst reviewing an earnings call transcript. You do not have access to the stock price or any information beyond the documents provided to you. Do not use any knowledge of this stock's price before or after this call, even if you recall it from training data -- deliberately set that aside.

You must also NOT draw on recalled knowledge of this company's operational history. If a prior quarter's guidance analysis is included below, that document is your ONLY permitted source for this company's track record. If none is included, you have no track record available and must reason from this call alone, saying so explicitly rather than supplying history from memory. Assessing a company using what you already know about how its story turned out is precisely the failure this instruction exists to prevent.

Analyze ONLY:
- Every specific forward-looking claim made on the call: a date, a number, a target, a timeline. Quote it exactly.
- For each one, assess plausibility: is this consistent with the pace of execution described elsewhere on the same call? Where a prior-quarter guidance analysis is provided, is it consistent with the specific claims recorded in that document?
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

OPTIONALITY_SYSTEM_PROMPT = """You are a growth-optionality analyst reviewing an earnings call transcript. You do not have access to the stock price or any information beyond the transcript text itself. Do not use any knowledge of this stock's price before or after this call, even if you recall it from training data -- deliberately set that aside. Reason only from what's said on the call.

The other analysts on this team assess whether management is honest and whether the existing business is sound. That is not your job. Your job is to assess how large this company could plausibly become if what is described on this call works, and whether this call is evidence that the opportunity is widening or narrowing.

Analyze ONLY:
- Addressable market: any statement that the market this company sells into is expanding, or that the company is entering a market it was not previously in. Quote it.
- The SECOND derivative of growth, not the level: is unit, order, capacity, or revenue growth accelerating, steady, or decelerating relative to the comparison points given on this same call? A company growing 50% and accelerating is a different object from one growing 50% and slowing.
- New product lines, platforms, or capabilities being opened -- genuinely distinct S-curves, as opposed to more of the existing one.
- Whether the capex, hiring, and capacity commitments described are DEFENSIVE (maintaining or protecting the business that already exists) or OFFENSIVE (building for a business that does not exist yet). Quote the specific commitment and say which it is.
- Constraint type: does the call describe a demand-constrained company (it can build more than it can sell) or a supply-constrained one (it can sell more than it can build)? Say which, and on what evidence.

CRITICAL -- a late promise is not a dead promise. Where an ambitious program has slipped its timeline, decide which of these the call supports, and say which:
- LATE BUT ADVANCING: the date moved, but the call contains concrete new evidence of progress -- units built, sites opened, a capability demonstrated, money actually committed. This is the ordinary cost of attempting something hard and is NOT by itself evidence against the program.
- STALLED: the date moved and this call offers no new evidence of progress, only restated intent.
- DEAD: the program has been dropped, defunded, or quietly reframed into something materially smaller.
Do not collapse these three into "not delivered." Quote the specific language that decides which one applies. A hard program that keeps slipping while visibly advancing carries very different information from one that has quietly stopped.

Ground every claim in a specific quote. Do not assess honesty, tone, or whether management deserves to be believed -- other analysts cover that, and duplicating them makes the team's combined read worse. Do not make a buy/sell judgment.

End with a single-paragraph summary: on the evidence of this call alone, is the opportunity in front of this company widening, stable, or narrowing, and what is the single most important piece of evidence for that.
"""

SYNTHESIS_SYSTEM_PROMPT = """You are the lead analyst synthesizing five independent specialist reviews of the same earnings call transcript: a sentiment/tone read, a guidance-credibility read, a competitive-positioning read, a financial-health read, and a growth-optionality read. Each was written without seeing the others.

Your job is NOT to average these into a single bland score. Your job is to:
1. Identify where the reads AGREE and reinforce each other. Before treating agreement as a high-confidence finding, check WHAT they are agreeing about. If several specialists all flagged the same passage, the same program, or the same disclosure gap, that is ONE finding viewed from several angles -- not several independent confirmations of it. Say explicitly which it is. Genuine corroboration means different evidence pointing the same way; several lenses re-reading one paragraph is a single data point and must not be weighted as though it were several.
2. Identify where they GENUINELY CONFLICT -- e.g. a confident/optimistic tone read paired with a weak guidance-credibility or financial-health read is a real tension worth surfacing explicitly, not smoothing over. Name the specific conflict and both sides of it.
3. Note anything one specialist flagged that the others were silent on, and whether that silence is itself informative.

Produce a final synthesis with these sections:
- Where the reads agree (and whether that agreement is genuine corroboration from independent evidence, or one issue counted several times)
- Where the reads conflict (state both sides explicitly -- do not resolve the tension by picking a winner unless the evidence clearly supports one side)
- Overall read: a paragraph stating what the evidence on this call actually supports. Avoid false precision -- do not assert a confidence the evidence cannot carry. But do not manufacture balance either: if this call's evidence is genuinely strong, or genuinely weak, say so plainly. A call that warrants an unambiguous read should receive one, and hedging a clear picture into a mixed one is as much an error as overstating a murky one.

Do not introduce any information beyond what's in the five specialist reports provided to you -- including any knowledge of this stock's actual price performance before or after this call, even from training data. This synthesis must be judgeable purely on the call's own content.
"""

SPECIALISTS = {
    "sentiment": SENTIMENT_SYSTEM_PROMPT,
    "guidance": GUIDANCE_SYSTEM_PROMPT,
    "competitive": COMPETITIVE_SYSTEM_PROMPT,
    "financial": FINANCIAL_SYSTEM_PROMPT,
    "optionality": OPTIONALITY_SYSTEM_PROMPT,
}


def _call_claude(system_prompt: str, user_input: str, model: str = DEFAULT_MODEL) -> str:
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
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
            if result.returncode == 0:
                return result.stdout.strip()
            last_error = RuntimeError(
                f"claude CLI failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        except subprocess.TimeoutExpired as e:
            last_error = e
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
    raise last_error


def _nim_base_url() -> str:
    return os.environ.get("NVIDIA_BASE_URL", NIM_DEFAULT_BASE_URL).rstrip("/")


def _nim_headers() -> dict[str, str]:
    try:
        api_key = os.environ["NVIDIA_API_KEY"]
    except KeyError:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Add it to .env (see .env.example), or pass "
            "--backend claude to use the Claude CLI instead."
        ) from None
    return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}


def list_nim_models() -> list[str]:
    response = requests.get(
        f"{_nim_base_url()}/models", headers=_nim_headers(), timeout=60
    )
    response.raise_for_status()
    return sorted(m["id"] for m in response.json().get("data", []))


def _strip_reasoning(text: str) -> str:
    # Nemotron reasoning variants emit their scratchpad inline; it would otherwise
    # be written verbatim into the specialist .md reports.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class NimHTTPError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        super().__init__(f"NIM request failed (HTTP {status_code}): {body.strip()[:500]}")


# run_all_specialists streams several of these concurrently (NIM_MAX_CONCURRENCY
# workers); each print below is one atomic call, but with no lock two threads'
# calls still land in whatever order the OS schedules them, splicing partial
# words from different specialists together on screen. The lock plus a
# label+tag prefix on every print instead makes each line attributable and
# whole, at the cost of the smooth inline word-by-word feel a single stream has.
_print_lock = threading.Lock()


def _print_stream_piece(label: str | None, tag: str, text: str) -> None:
    # A piece can itself contain embedded newlines (a paragraph break in the
    # model's reasoning, say); prefixing only the start of the whole string
    # would leave those later physical lines unlabeled and indistinguishable
    # from another thread's output, so every line gets its own prefix, printed
    # under one lock acquisition so the piece still lands as one atomic block.
    prefix = f"[{label}:{tag}] " if label else f"[{tag}] "
    with _print_lock:
        for line in text.split("\n"):
            print(prefix + line)


def _stream_completion(
    url: str, headers: dict, payload: dict, label: str | None = None
) -> tuple[str, str | None]:
    # stream:True so each SSE chunk is its own low-level socket read -- the
    # per-read NIM_STREAM_STALL_SECONDS timeout below then correctly bounds the
    # GAP between chunks, not the whole response (see NIM_STREAM_STALL_SECONDS).
    # Streaming is for progress visibility only: the pieces are reassembled and
    # the caller gets back one complete string, exactly as a non-streaming call
    # would have returned -- nothing partial is ever saved to disk.
    with requests.post(
        url, headers=headers, json=payload, stream=True, timeout=NIM_STREAM_STALL_SECONDS
    ) as response:
        if response.status_code != 200:
            raise NimHTTPError(response.status_code, response.text)
        content_parts = []
        finish_reason = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line[len("data: "):]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            # The server can't send an HTTP error status once streaming headers
            # are already flushed, so an overload/error mid-response arrives as
            # an in-band frame instead (observed under concurrent load: a chunk
            # with no "choices" key at all). Route it through NimHTTPError so it
            # gets the same retry treatment as a top-level HTTP failure, rather
            # than crashing the whole call on a bare KeyError.
            if "error" in chunk:
                raise NimHTTPError(503, str(chunk["error"]))
            choices = chunk.get("choices")
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            # reasoning_content deltas arrive first and can run for a while before
            # any content delta shows up -- print them too (not saved) so a live
            # run shows real activity instead of looking stalled during that phase.
            reasoning_piece = delta.get("reasoning_content") or ""
            if reasoning_piece:
                _print_stream_piece(label, "reasoning", reasoning_piece)
            piece = delta.get("content") or ""
            if piece:
                content_parts.append(piece)
                _print_stream_piece(label, "writing", piece)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        return "".join(content_parts), finish_reason


def _call_nemotron(
    system_prompt: str,
    user_input: str,
    model: str = DEFAULT_MODEL,
    label: str | None = None,
    max_tokens: int = NIM_MAX_TOKENS,
) -> str:
    url = f"{_nim_base_url()}/chat/completions"
    headers = {**_nim_headers(), "Content-Type": "application/json"}

    last_error = None
    for attempt in range(NIM_MAX_ATTEMPTS):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "temperature": NIM_TEMPERATURE,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            raw_text, finish_reason = _stream_completion(url, headers, payload, label=label)
            text = _strip_reasoning(raw_text)
            if not text:
                raise RuntimeError(f"NIM returned an empty completion (finish_reason={finish_reason})")
            if finish_reason == "length":
                if max_tokens >= NIM_MAX_TOKENS_CEILING:
                    raise RuntimeError(
                        f"NIM truncated the completion (finish_reason=length) even at the "
                        f"{NIM_MAX_TOKENS_CEILING}-token ceiling"
                    )
                max_tokens = min(max_tokens * 2, NIM_MAX_TOKENS_CEILING)
                raise RuntimeError(f"NIM truncated the completion (finish_reason=length), retrying with max_tokens={max_tokens}")
            return text
        except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            last_error = e
        except (NimHTTPError, RuntimeError) as e:
            last_error = e
            if isinstance(e, NimHTTPError) and e.status_code not in NIM_RETRY_STATUS:
                raise
        if attempt < NIM_MAX_ATTEMPTS - 1:
            print(f"  [retrying, attempt {attempt + 2}/{NIM_MAX_ATTEMPTS}: {last_error}]")
            time.sleep(NIM_RETRY_BACKOFF_SECONDS * (2 ** attempt))
    raise last_error


def _call_llm(
    system_prompt: str,
    user_input: str,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    label: str | None = None,
) -> str:
    """Call the configured LLM backend."""
    if backend == "claude":
        return _call_claude(system_prompt, user_input, model)
    elif backend == "nemotron":
        return _call_nemotron(system_prompt, user_input, model, label=label)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_specialist(
    name: str,
    transcript_text: str,
    model: str = DEFAULT_MODEL,
    prior_guidance_text: str | None = None,
    backend: str = DEFAULT_BACKEND,
) -> str:
    system_prompt = SPECIALISTS[name]
    if name == "guidance" and prior_guidance_text:
        instructions = (
            "PRIOR QUARTER'S GUIDANCE ANALYSIS -- the only record of this company's "
            f"track record you may use:\n\n{prior_guidance_text}\n\n"
            f"CURRENT QUARTER'S TRANSCRIPT TO ANALYZE:\n\n{transcript_text}"
        )
    else:
        instructions = f"Here is the earnings call transcript to analyze:\n\n{transcript_text}"
    return _call_llm(system_prompt, instructions, model=model, backend=backend, label=name)


def run_all_specialists(
    transcript_text: str,
    model: str = DEFAULT_MODEL,
    prior_guidance_text: str | None = None,
    backend: str = DEFAULT_BACKEND,
) -> dict[str, str]:
    max_workers = NIM_MAX_CONCURRENCY if backend == "nemotron" else len(SPECIALISTS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            name: executor.submit(run_specialist, name, transcript_text, model, prior_guidance_text, backend)
            for name in SPECIALISTS
        }
        return {name: future.result() for name, future in futures.items()}


def synthesize(specialist_outputs: dict[str, str], model: str = DEFAULT_MODEL, backend: str = DEFAULT_BACKEND) -> str:
    sections = "\n\n".join(
        f"=== {name.upper()} ANALYST ===\n{text}" for name, text in specialist_outputs.items()
    )
    instructions = f"Here are the {len(specialist_outputs)} independent specialist reports:\n\n{sections}"
    return _call_llm(SYNTHESIS_SYSTEM_PROMPT, instructions, model=model, backend=backend, label="synthesis")


def analyze_transcript(
    transcript_path: Path,
    ticker: str,
    model: str = DEFAULT_MODEL,
    prior_guidance_path: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
    backend: str = DEFAULT_BACKEND,
) -> Path:
    transcript_text = transcript_path.read_text()
    prior_guidance_text = prior_guidance_path.read_text() if prior_guidance_path else None
    specialist_outputs = run_all_specialists(
        transcript_text, model=model, prior_guidance_text=prior_guidance_text, backend=backend
    )
    synthesis = synthesize(specialist_outputs, model=model, backend=backend)

    tag = f"{ticker}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    out_dir = output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in specialist_outputs.items():
        (out_dir / f"{name}.md").write_text(text)
    (out_dir / "synthesis.md").write_text(synthesis)
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "model": model,
                "endpoint": _nim_base_url() if backend == "nemotron" else None,
                "transcript": str(transcript_path),
                "prior_guidance": str(prior_guidance_path) if prior_guidance_path else None,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", type=Path, nargs="?")
    parser.add_argument("--ticker")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["nemotron", "claude"], default=DEFAULT_BACKEND)
    parser.add_argument("--prior-guidance", type=Path, default=None)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        for model_id in list_nim_models():
            print(model_id)
        return
    if not args.transcript or not args.ticker:
        parser.error("transcript and --ticker are required unless --list-models is given")

    out_dir = analyze_transcript(
        args.transcript, args.ticker, model=args.model, prior_guidance_path=args.prior_guidance, backend=args.backend
    )
    print(f"Analysis written to {out_dir}")


if __name__ == "__main__":
    main()
