import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.earnings_call_analyst import DEFAULT_BACKEND, DEFAULT_MODEL, _call_llm

from anonymize_transcripts import GENERATED_CONFIG_DIR, TRANSCRIPT_DIR
from suggest_anonymize_config import gather_candidates

SNIPPET_RADIUS = 60
CANDIDATE_LIMIT = 120

SYSTEM_PROMPT = """You help anonymize earnings-call transcripts so a reader cannot tell which company is speaking. You are given the company's name, its ticker, and a list of candidate terms pulled from its transcripts, each with how often it appears and one snippet of context.

Select the terms that would let a reader identify THIS company: its own products, platforms, technologies, programs, brands, subsidiaries, events, and internal jargon that is specific to it.

Do NOT select:
- generic industry or technical terms (for example GPU, cloud, AI, ERP, LLM)
- geographies, macro or financial terms, months, or acronyms in general use
- third-party companies of any kind: customers, partners, competitors, and analysts' firms

Group names that refer to one thing (a product family and its abbreviations, a program and its event name) under one entry with a short canonical name. Keep successive generations of a product as SEPARATE entries so the sequence stays visible after anonymizing.

Every alias you output must be copied exactly from the candidate list. Do not add terms from your own knowledge.

Output ONLY a JSON object (no markdown fences, no commentary):
{"programs": {"canonical name": ["alias", "alias"], ...}}
"""


def _snippets(files: list[Path], terms: list[str]) -> dict[str, str]:
    text = "\n".join(f.read_text() for f in files)
    found = {}
    for term in terms:
        match = re.search(r"\b" + re.escape(term) + r"\b", text)
        if match:
            start = max(0, match.start() - SNIPPET_RADIUS)
            found[term] = " ".join(text[start : match.end() + SNIPPET_RADIUS].split())
    return found


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


def generate_config(ticker: str, files: list[Path], model: str, backend: str) -> dict:
    company_name, ranked = gather_candidates(ticker, files, top=CANDIDATE_LIMIT)
    if not ranked:
        raise RuntimeError(f"no candidate terms found for {ticker}")
    snippets = _snippets(files, [t for t, _, _ in ranked])
    lines = [
        f"- {term} (x{count}, {quarters} quarters): ...{snippets.get(term, '')}..."
        for term, count, quarters in ranked
    ]
    prompt = f"Company: {company_name}\nTicker: {ticker}\n\nCandidate terms:\n" + "\n".join(lines)
    response = _call_llm(SYSTEM_PROMPT, prompt, model=model, backend=backend, label=f"{ticker}:anon-config")
    programs = _parse_json(response).get("programs", {})

    corpus = "\n".join(f.read_text() for f in files).lower()
    kept: dict[str, list[str]] = {}
    for canonical, aliases in programs.items():
        valid = [a for a in aliases if isinstance(a, str) and a.lower() in corpus]
        if valid:
            kept[canonical] = valid

    suffixes = ["", " Inc.", " Corporation", " Corp", " Holdings"]
    company = sorted({company_name + s for s in suffixes} | {ticker}, key=len, reverse=True)
    return {"company": company, "programs": kept}


def main():
    parser = argparse.ArgumentParser(
        description="Generate an anonymization config for a ticker with one LLM call over its recurring "
        "terms. The result is saved as JSON, where it can be reviewed and edited."
    )
    parser.add_argument("ticker")
    parser.add_argument("--input-dir", type=Path, default=TRANSCRIPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=GENERATED_CONFIG_DIR)
    parser.add_argument("--backend", choices=["nemotron", "claude"], default=DEFAULT_BACKEND)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    ticker = args.ticker.upper()

    files = sorted(args.input_dir.glob(f"{ticker}_*Q*.txt"))
    if not files:
        raise SystemExit(f"no transcripts for {ticker} in {args.input_dir}")
    model = args.model or (DEFAULT_MODEL if args.backend == "nemotron" else "sonnet")

    config = generate_config(ticker, files, model, args.backend)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{ticker}.json"
    out_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"{ticker}: {len(config['programs'])} programs -> {out_path}")
    for canonical, aliases in config["programs"].items():
        print(f"  {canonical}: {', '.join(aliases)}")


if __name__ == "__main__":
    main()
