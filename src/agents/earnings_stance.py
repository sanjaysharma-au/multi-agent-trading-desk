import argparse
import json
import subprocess
from pathlib import Path

CLAUDE_TIMEOUT_SECONDS = 600
DEFAULT_MODEL = "sonnet"

STANCE_SYSTEM_PROMPT = """You are a stock-stance analyst. You are given a finished four-analyst synthesis of an earnings call, plus the stock's actual closing price on the day of the call.

CRITICAL: you must not use any knowledge of what actually happened to this company or its stock price after this specific call. Reason only from what's in the synthesis and the stated current price, as if you were an analyst on the date of this call with no ability to see the future. If you have prior knowledge of this company's subsequent history, deliberately set it aside for this task.

Produce a directional stance -- BULLISH, NEUTRAL, or BEARISH -- based solely on the synthesis provided.

Then give three scenarios as PERCENTAGE moves from the given current price, over a 12-month horizon:
- Bull case: the % move if the most credible positive claims in the synthesis play out
- Base case: the % move under the synthesis's most balanced, likely read
- Bear case: the % move if the biggest identified risks or unresolved tensions materialize

Ground every scenario in specific reasoning from the synthesis -- name what would need to happen for the bull case and what would need to go wrong for the bear case.

End your response with a single JSON object on its own line (no markdown fences) with exactly these keys: {"stance": "bullish|neutral|bearish", "bull_pct": <number>, "base_pct": <number>, "bear_pct": <number>}
"""


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


def assess_stance(synthesis_text: str, current_price: float, model: str = DEFAULT_MODEL) -> tuple[str, dict]:
    instructions = f"Current stock price at the time of this call: ${current_price:.2f}\n\nSynthesis report:\n\n{synthesis_text}"
    response = _call_claude(STANCE_SYSTEM_PROMPT, instructions, model=model)

    json_line = None
    for line in reversed(response.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line
            break
    scores = json.loads(json_line) if json_line else {}
    return response, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("synthesis_path", type=Path)
    parser.add_argument("current_price", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    full_text, scores = assess_stance(args.synthesis_path.read_text(), args.current_price, model=args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(full_text)
    print(f"Stance analysis written to {args.output}")
    print(scores)


if __name__ == "__main__":
    main()
