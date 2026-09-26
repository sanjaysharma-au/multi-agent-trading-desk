"""Scrub company/product/person identity and absolute dates from earnings-call
transcripts, so a model's stated analysis can be compared identified-vs-anonymized
as a leakage measurement (see docs plan: "the gap *is* the leak measurement").

Not a general-purpose NER anonymizer -- deliberately narrow and auditable:
- Company + ticker aliases and named program/product aliases are hand-curated per
  ticker (CONFIG below), since generic NER both over- and under-scrubs proper
  nouns specific to a transcript's domain.
- Speaker names are DISCOVERED structurally (a name alone on its own line before
  its remarks -- verified across HYLN and TSLA samples), but then substituted
  EVERYWHERE they appear in the text, not just on that structural line -- an
  exec is routinely named again in surrounding prose ("we have Thomas Healy, our
  CEO"), and a line-anchored-only substitution leaves those instances exposed.
- Years are shifted to an offset RELATIVE to the transcript's own quarter
  ("Y+0" = the quarter this call is reporting, "Y+2" = two years after), so a
  dated promise still reads as dated without exposing the real calendar year a
  model could anchor its trained knowledge on.

Substitution order matters: every alias (company, all programs, all people) is
combined into ONE regex per file, sorted longest-alias-first, and applied in a
SINGLE pass. Applying aliases one at a time in a loop re-scans already-
substituted text on each later iteration -- a short alias that happens to be a
substring of an earlier alias (e.g. "ERX" inside "Hypertruck ERX") then matches
again inside the placeholder just written, producing corrupted nested output
like "Program [Hypertruck Program [ERX]]". A single combined pass can't do that
because each source position is consumed by at most one match.

Known residual leakage channel, deliberately not addressed: sell-side analyst
names' employer firms (Goldman Sachs, Cantor Fitzgerald, ...) are left intact.
Which banks cover a company is a weaker identity signal than the company's own
name and scrubbing every third-party firm mentioned on every call is a much
larger, more error-prone undertaking than this script's scope. If the
anonymized-vs-identified gap this produces turns out non-trivial, revisit.
"""

import argparse
import json
import re
from pathlib import Path

TRANSCRIPT_DIR = Path("data/earnings_calls")
DEFAULT_OUTPUT_DIR = Path("data/earnings_calls_anon")

# All matching is case-insensitive (see anonymize_transcript), so this list
# does not need separate-case duplicates ("KARNO" / "Karno") -- one entry
# catches all. A regular "+s" plural of every program alias is added
# automatically ("\bHypertruck\b" does not match inside "Hypertrucks"), but
# irregular plurals (Gigafactory -> Gigafactories) need their own entries.
CONFIG = {
    "HYLN": {
        "company": ["Hyliion Holdings Corp", "Hyliion Holdings", "Hyliion", "HYLN"],
        "programs": {
            "Hypertruck ERX": ["Hypertruck ERX", "Hypertruck", "Hypertrucks", "ERX", "ERXs"],
            "Hybrid e-Axle": ["Hybrid e-Axle", "e-Axle", "Hybrid system", "Hybrid powertrain"],
            "Karno Generator": ["Karno"],
        },
    },
    "PLTR": {
        "company": ["Palantir Technologies Inc.", "Palantir Technologies", "Palantirians", "Palantirian", "Palantir", "PLTR"],
        "programs": {
            "Gotham": ["Gotham"],
            "Foundry": ["FoundryCon", "Foundry for Builders", "Foundry"],
            "Apollo": ["Apollo"],
            "AIP": ["AIPCon", "AIP"],
            "Maven": ["Maven Smart System", "Project Maven", "Maven"],
            "Warp Speed": ["Warp Speed"],
            "Ontology": ["Ontology"],
            "MetaConstellation": ["MetaConstellation", "Meta-Constellation"],
            "Skykit": ["Skykit"],
            "TITAN": ["TITAN"],
            "Army Vantage": ["Army Vantage", "Vantage"],
            "Tiberius": ["Tiberius"],
            "Bootcamp": ["Bootcamp", "Bootcamps"],
        },
    },
    "NVDA": {
        "company": ["NVIDIA Corporation", "NVIDIA Corp", "NVIDIA", "NVDA"],
        "programs": {
            "GeForce": ["GeForce", "GTX", "RTX"],
            "Tegra": ["Tegra", "Xavier", "Orin", "Thor", "Jetson", "Shield"],
            "Quadro": ["Quadro"],
            "Tesla": ["Tesla"],
            "DRIVE": ["NVIDIA DRIVE"],
            "DGX": ["DGX", "HGX", "MGX"],
            "CUDA": ["CUDA"],
            "Omniverse": ["Omniverse", "Cosmos"],
            "Fermi": ["Fermi"],
            "Kepler": ["Kepler"],
            "Maxwell": ["Maxwell"],
            "Pascal": ["Pascal"],
            "Volta": ["Volta"],
            "Turing": ["Turing"],
            "Ampere": ["Ampere"],
            "Ada": ["Ada Lovelace", "Lovelace"],
            "Hopper": ["Hopper"],
            "Blackwell": ["Blackwell"],
            "Rubin": ["Vera Rubin", "Rubin"],
            "Vera": ["Vera"],
            "Grace": ["Grace"],
            "NVLink": ["NVLink"],
            "BlueField": ["BlueField"],
            "Spectrum-X": ["Spectrum-X"],
            "Mellanox": ["Mellanox"],
            "Icera": ["Icera"],
            "GTC": ["GTC"],
            "Titan": ["Titan"],
            "K80": ["K80"],
            "P100": ["P100"],
            "V100": ["V100"],
            "T4": ["T4"],
            "A100": ["A100"],
            "H100": ["H100"],
            "H200": ["H200"],
            "B200": ["B200", "GB200"],
            "L40": ["L40"],
            "Nemotron": ["Nemotron", "NeMo"],
        },
    },
    "TSLA": {
        "company": ["Tesla Motors, Inc.", "Tesla Motors", "Tesla, Inc.", "Tesla Inc.", "Tesla", "TSLA"],
        "programs": {
            "Model 3": ["Model 3"],
            "Model S": ["Model S"],
            "Model X": ["Model X"],
            "Model Y": ["Model Y"],
            "Cybertruck": ["Cybertruck", "Cybertrucks"],
            "Full Self-Driving": ["Full Self-Driving", "FSD"],
            "Autopilot": ["Autopilot"],
            "Gigafactory": ["Gigafactory", "Gigafactories", "Giga Nevada", "Giga Texas", "Giga Berlin", "Giga Shanghai"],
            "Semi": ["Tesla Semi", "the Semi"],
            "Roadster": ["Roadster", "Roadsters"],
            "Powerwall": ["Powerwall", "Powerwalls"],
            "Solar Roof": ["Solar Roof", "Solar Roofs"],
            "Dojo": ["Dojo"],
            "Optimus": ["Optimus"],
            "Supercharger": ["Supercharger", "Superchargers"],
        },
    },
}

GENERATED_CONFIG_DIR = Path("data/anonymize_configs")


def _generated_config_path(ticker: str) -> Path:
    return GENERATED_CONFIG_DIR / f"{ticker}.json"


def has_config(ticker: str) -> bool:
    return ticker in CONFIG or _generated_config_path(ticker).exists()


def get_config(ticker: str) -> dict:
    if ticker in CONFIG:
        return CONFIG[ticker]
    return json.loads(_generated_config_path(ticker).read_text())


SPEAKER_LINE_RE = re.compile(r"^[A-Z][a-zA-Z.'-]+(?: [A-Z][a-zA-Z.'-]+){1,3}$", re.MULTILINE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _quarter_year(quarter_label: str) -> int:
    return int(re.match(r"(\d{4})Q(\d)", quarter_label).group(1))


def discover_people(files: list[Path]) -> dict[str, str]:
    """First pass over every file in the ticker's set: collect every structural
    speaker name before any substitution happens, so a person introduced in an
    early quarter is still recognized (and scrubbed) if named again in prose in
    a later one.

    A capitalized line alone on its own line is only accepted as a speaker if the
    next line is a title ("CEO, Company") or the same line recurs elsewhere;
    a one-off heading such as "Each Grace Blackwell NVLink" is not a person."""
    counts: dict[str, int] = {}
    titled: set[str] = set()
    for f in files:
        lines = f.read_text().splitlines()
        for i, line in enumerate(lines):
            if not SPEAKER_LINE_RE.fullmatch(line) or line.endswith("."):
                continue
            counts[line] = counts.get(line, 0) + 1
            if i + 1 < len(lines) and "," in lines[i + 1]:
                titled.add(line)
    person_map: dict[str, str] = {}
    for name, count in counts.items():
        if name in titled or count >= 2:
            person_map[name] = f"Person {len(person_map) + 1}"
    return person_map


def common_lowercase_words(files: list[Path]) -> set[str]:
    words: set[str] = set()
    for f in files:
        words.update(re.findall(r"\b[a-z][a-z'-]*\b", f.read_text()))
    return words


def _build_alias_table(
    ticker: str, person_map: dict[str, str], common_words: frozenset[str] | set[str] = frozenset()
) -> list[tuple[str, str]]:
    """One flat (alias, placeholder) list -- company, all program aliases, all
    people -- consumed as a single combined regex so no alias can match inside
    a placeholder another alias just produced.

    Program placeholders must NOT embed the real canonical name (an earlier
    version used f"Program [{canonical}]", e.g. "Program [Karno Generator]" --
    that string still contains "Karno Generator" verbatim and anonymizes
    nothing). Assign a numbered label instead, in CONFIG's definition order so
    it's deterministic and stable across runs."""
    cfg = get_config(ticker)
    table = [(a, "Company A") for a in cfg["company"]]
    for i, (canonical, aliases) in enumerate(cfg["programs"].items(), start=1):
        for a in aliases:
            table.append((a, f"Program {i}"))
            if not a.lower().endswith("s"):
                table.append((a + "s", f"Program {i}"))
    table.extend(person_map.items())
    # People are routinely addressed by surname alone ("Mr. Healy") or first
    # name alone ("Thanks, Thomas" -- earnings-call Q&A is informal) elsewhere
    # in the transcript, not just by the full name their speaker-line used.
    # Map both to the same placeholder.
    # Known limitation: if two speakers share a first or last name, their bare-
    # name mentions collapse onto whichever placeholder's table entry sorts
    # first (full-name and structural speaker-line mentions stay correct).
    # A bare name that also appears in lowercase elsewhere is an ordinary word
    # ("Cash", "Mark"); mapping it would scrub the word everywhere it is used.
    for full_name, placeholder in list(person_map.items()):
        parts = full_name.split(" ")
        for bare in {parts[0], parts[-1]}:
            if bare.lower() not in common_words:
                table.append((bare, placeholder))
    return table


def anonymize_transcript(
    text: str,
    ticker: str,
    quarter_label: str,
    person_map: dict[str, str],
    common_words: frozenset[str] | set[str] = frozenset(),
) -> str:
    table = _build_alias_table(ticker, person_map, common_words)
    table.sort(key=lambda pair: len(pair[0]), reverse=True)
    # Case-insensitive: the same real name/term can appear in different casing
    # ("KARNO" vs "Karno" vs a stray lowercase mention) and all must map to the
    # same placeholder. The lookup is keyed on the lowercased match so casing
    # in the source text doesn't matter at substitution time.
    #
    # Plain \b uses Unicode \w, which some of these scraped transcripts defeat:
    # a mangled apostrophe ("Elon\xe2\x80\x99s" -> "Elonâs" after a bad decode)
    # counts as a word character, so \b sees no boundary between "Elon" and the
    # corrupted possessive suffix and the alias silently fails to match. An
    # explicit ASCII-letter/digit boundary catches the real case (start/end of
    # an actual word) without depending on what non-ASCII punctuation follows.
    start_boundary = r"(?<![A-Za-z0-9])"
    end_boundary = r"(?![A-Za-z0-9])"
    pattern = re.compile(
        start_boundary + r"(?:" + "|".join(re.escape(a) for a, _ in table) + r")" + end_boundary,
        re.IGNORECASE,
    )
    lookup: dict[str, str] = {}
    for alias, placeholder in table:
        lookup.setdefault(alias.lower(), placeholder)

    text = pattern.sub(lambda m: lookup[m.group(0).lower()], text)
    text = YEAR_RE.sub(lambda m: _year_offset(m.group(0), quarter_label), text)
    return text


def _year_offset(year_str: str, quarter_label: str) -> str:
    offset = int(year_str) - _quarter_year(quarter_label)
    return f"Y+{offset}" if offset >= 0 else f"Y{offset}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--input-dir", type=Path, default=TRANSCRIPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not has_config(args.ticker):
        raise SystemExit(
            f"No anonymization config for {args.ticker}: add it to CONFIG or run "
            f"generate_anonymize_config.py {args.ticker}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob(f"{args.ticker}_*Q*.txt"))
    print(f"{len(files)} transcripts found for {args.ticker}")

    person_map = discover_people(files)
    common_words = common_lowercase_words(files)
    print(f"{len(person_map)} distinct people discovered across {args.ticker}'s transcript set")

    for f in files:
        label = f.stem.replace(f"{args.ticker}_", "")
        anon = anonymize_transcript(f.read_text(), args.ticker, label, person_map, common_words)
        out_path = args.output_dir / f.name
        out_path.write_text(anon)
        print(f"[{label}] -> {out_path}")


if __name__ == "__main__":
    main()
