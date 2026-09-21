import argparse
import subprocess
from pathlib import Path

CLAUDE_TIMEOUT_SECONDS = 600
DEFAULT_MODEL = "sonnet"

TRAJECTORY_SYSTEM_PROMPT = """You are a quarter-over-quarter execution tracker. You are given (1) the specific forward-looking guidance and promises extracted from a company's PREVIOUS earnings call, and (2) the full transcript of the CURRENT, more recent earnings call.

You do not have access to the stock price and must not use any knowledge of how the stock performed before, during, or after either call -- even if you recall it from training data, deliberately set it aside. Judge execution and credibility purely from what these two documents say, not from how the market reacted or how the story ended.

Your job is to check whether the current call confirms that promises from the previous call were kept. For each specific, checkable claim in the previous guidance report:
1. Quote the original promise.
2. Search the current transcript for any related discussion, and quote it.
3. Classify the outcome as one of: DELIVERED, MISSED, REVISED (say how -- softened, delayed, reframed), or NOT ADDRESSED (silence on a previously prominent promise is itself informative -- note it).

Do not use any information beyond what's in these two documents -- no outside knowledge of what actually happened to this company after the current call's date.

End with a verdict: is this company's execution trajectory IMPROVING, STABLE, or DETERIORATING this quarter, based on the ratio of delivered vs. missed vs. quietly revised promises, and whether the new guidance given this quarter is more or less credible than what preceded it.
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


def assess_trajectory(prior_guidance_text: str, current_transcript_text: str, model: str = DEFAULT_MODEL) -> str:
    instructions = (
        f"PREVIOUS QUARTER'S GUIDANCE ANALYSIS:\n\n{prior_guidance_text}\n\n"
        f"CURRENT QUARTER'S FULL TRANSCRIPT:\n\n{current_transcript_text}"
    )
    return _call_claude(TRAJECTORY_SYSTEM_PROMPT, instructions, model=model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prior_guidance_path", type=Path)
    parser.add_argument("current_transcript_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    result = assess_trajectory(
        args.prior_guidance_path.read_text(),
        args.current_transcript_path.read_text(),
        model=args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(f"Trajectory analysis written to {args.output}")


if __name__ == "__main__":
    main()
