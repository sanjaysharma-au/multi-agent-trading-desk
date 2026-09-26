import argparse
import re
from collections import Counter
from pathlib import Path

from anonymize_transcripts import (
    SPEAKER_LINE_RE,
    TRANSCRIPT_DIR,
    common_lowercase_words,
    discover_people,
)

TERM_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")
STOPLIST = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "us", "u.s", "uk", "eu", "ceo", "cfo", "coo", "cto", "gaap", "non-gaap", "eps", "yoy", "qoq",
    "sec", "ebitda", "fcf", "ir", "ai", "ml", "it", "q1", "q2", "q3", "q4", "fy", "operator",
    "person", "company", "program", "thanks", "thank", "okay", "ok", "yes", "no", "hi", "hello",
    "good", "great", "well", "sure", "right", "so", "and", "but", "the", "our", "we", "you", "i",
}


def company_names_from_titles(files: list[Path]) -> Counter:
    counts: Counter = Counter()
    for f in files:
        lines = f.read_text().splitlines()
        for i, line in enumerate(lines[:-1]):
            if SPEAKER_LINE_RE.fullmatch(line) and not line.endswith(".") and "," in lines[i + 1]:
                counts[lines[i + 1].rsplit(",", 1)[1].strip()] += 1
    return counts


def candidate_terms(files: list[Path], excluded: set[str], common_words: set[str]) -> dict[str, tuple[int, int]]:
    total: Counter = Counter()
    quarters: Counter = Counter()
    for f in files:
        terms = TERM_RE.findall(f.read_text())
        total.update(terms)
        quarters.update(set(terms))
    result = {}
    for term, count in total.items():
        low = term.lower()
        if low in STOPLIST or low in common_words or low in excluded or len(term) < 2 or term.isdigit():
            continue
        result[term] = (count, quarters[term])
    return result


def gather_candidates(
    ticker: str, files: list[Path], min_count: int = 5, min_quarters: int = 3, top: int = 60
) -> tuple[str, list[tuple[str, int, int]]]:
    people = discover_people(files)
    person_tokens = {tok.lower() for name in people for tok in name.split()}
    common_words = common_lowercase_words(files)

    companies = company_names_from_titles(files)
    company_name = companies.most_common(1)[0][0] if companies else ticker
    company_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", company_name)}

    candidates = candidate_terms(files, person_tokens | company_tokens | {ticker.lower()}, common_words)
    ranked = sorted(
        ((t, c, q) for t, (c, q) in candidates.items() if c >= min_count and q >= min_quarters),
        key=lambda x: -x[1],
    )[:top]
    return company_name, ranked


def main():
    parser = argparse.ArgumentParser(
        description="Print a rule-based draft CONFIG entry for anonymize_transcripts.py. It cannot tell "
        "generic terms from company-specific ones; generate_anonymize_config.py adds that judgement."
    )
    parser.add_argument("ticker")
    parser.add_argument("--input-dir", type=Path, default=TRANSCRIPT_DIR)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--min-quarters", type=int, default=3)
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()
    ticker = args.ticker.upper()

    files = sorted(args.input_dir.glob(f"{ticker}_*Q*.txt"))
    if not files:
        raise SystemExit(f"no transcripts for {ticker} in {args.input_dir}")

    company_name, ranked = gather_candidates(ticker, files, args.min_count, args.min_quarters, args.top)
    print(f"{ticker}: {len(files)} transcripts")
    print(f"Company name from speaker titles: {company_name!r}")
    print("\nCandidate terms (count, quarters appearing in):")
    for term, count, quarters in ranked:
        print(f"  {term:<28} {count:>5} {quarters:>3}")

    aliases = [company_name + s for s in (" Inc.", " Corporation", " Corp", " Holdings")] + [company_name, ticker]
    print("\nDraft entry:\n")
    print(f'    "{ticker}": {{')
    print(f'        "company": {aliases!r},'.replace("'", '"'))
    print('        "programs": {')
    for term, _, _ in ranked:
        print(f'            "{term}": ["{term}"],')
    print("        },")
    print("    },")


if __name__ == "__main__":
    main()
