"""Local control server: serves the inflections report live, with per-ticker buttons that
start/stop the earnings-call moonshot pipeline (multi-agent-trading-desk) and poll its status.

Stdlib only, no new dependency. Same-origin by design: the report HTML, the JSON API, and the
subprocess control all live behind one http.server so the page's fetch() calls need no CORS setup.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .report_inflections import DEFAULT_MOONSHOT_DIR, _moonshot_payload, build_inflections_report

# tools/stock-screener -> tools -> multi-agent-trading-desk
REPO_ROOT = Path(__file__).resolve().parents[4]
PIPELINE = REPO_ROOT / "src" / "research" / "run_moonshot_pipeline.py"
PY = REPO_ROOT / ".venv" / "bin" / "python"
LOG_TAIL_CHARS = 4000

# Cache the built page across `serve` restarts (fetching fundamentals for every ticker takes
# ~20-30s). Invalidated automatically if any of the input CSVs, the moonshot scores, or this
# project's own code change - see _fingerprint(). Does NOT track changes inside --data-dir (the
# SEC/price cache): a fresh `moonshot scan`/`inflect` run changes the crossings/failures CSVs
# too, which already invalidates the cache, but editing only the cached SEC data without
# re-running inflect would not. Pass --rebuild to force a fresh build regardless.
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
CACHE_HTML = CACHE_DIR / "live_report.html"
CACHE_MANIFEST = CACHE_DIR / "live_report.manifest.json"
_FINGERPRINT_SOURCE_FILES = [
    Path(__file__), Path(__file__).with_name("cli.py"), Path(__file__).with_name("report.py"),
    Path(__file__).with_name("report_inflections.py"), Path(__file__).with_name("inflections_template.html"),
]


def _fingerprint(crossings_csv: Path | None, failures_csv: Path | None,
                  crossings_scores: Path | None, failures_scores: Path | None, moonshot_dir: Path) -> str:
    h = hashlib.sha256()
    for p in _FINGERPRINT_SOURCE_FILES:
        if p.exists():
            st = p.stat()
            h.update(f"{p.name}:{st.st_mtime_ns}".encode())
    for p in (crossings_csv, failures_csv, crossings_scores, failures_scores):
        if p and p.exists():
            st = p.stat()
            h.update(f"{p}:{st.st_mtime_ns}:{st.st_size}".encode())
    if moonshot_dir.exists():
        for f in sorted(moonshot_dir.glob("*_moonshot_v3.jsonl")):
            h.update(f"{f.name}:{f.stat().st_mtime_ns}".encode())
    return h.hexdigest()

BACKEND_ARGS = {
    "nemo": [],
    "claude": ["--backend", "claude", "--model", "sonnet", "--ledger-backend", "claude", "--ledger-model", "sonnet"],
}

CHAT_SYSTEM_PROMPT = """You are a research analyst helping someone understand a specific stock, inside a \
stock-screener tool. You are given below the ticker's fundamentals-crossing data and the earnings-call \
moonshot pipeline's per-quarter scores (proximity, momentum, ambition, runway, tier, stage) with the \
scorer's own one-line rationale for each quarter. Both are blind to price and to how the story turned out.

Answer only from this context and clearly-labeled general knowledge; say so when you are not sure. Keep \
answers short and direct - a few sentences unless the question asks for more. Do not fetch anything, run \
any tool, or make up figures not in the context or your knowledge."""

CLAUDE_CHAT_TIMEOUT_SECONDS = 120


def _build_ticker_context(ticker: str, crossings_csv: Path | None, failures_csv: Path | None,
                           crossings_scores: Path | None, failures_scores: Path | None,
                           moonshot_dir: Path) -> str:
    lines = [f"Ticker: {ticker}"]
    for csv_path, scores_path in ((crossings_csv, crossings_scores), (failures_csv, failures_scores)):
        if not csv_path or not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError):
            continue
        match = df[df["ticker"] == ticker]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        if row.get("name"):
            lines.append(f"Name: {row['name']}")
        for k, v in row.items():
            if k in ("ticker", "name") or pd.isna(v):
                continue
            lines.append(f"{k}: {v}")
        if scores_path and scores_path.exists():
            try:
                srow = pd.read_csv(scores_path)
                srow = srow[srow["ticker"] == ticker]
            except (OSError, pd.errors.ParserError):
                srow = None
            if srow is not None and not srow.empty:
                s = srow.iloc[0]
                lines.append(f"Fundamentals score (Nemotron, blind to price): {s.get('score')} - {s.get('rationale', '')}")
        break

    rows = _moonshot_payload([ticker], moonshot_dir).get(ticker, [])
    if rows:
        lines.append("\nEarnings-call moonshot pipeline, one line per scored quarter "
                      "(quarter, proximity, momentum, ambition, runway, tier, stage, rationale):")
        for r in rows:
            lines.append(f"- {r[0]}: proximity={r[1]} momentum={r[2]} ambition={r[3]} runway={r[4]} "
                          f"tier={r[5]} stage={r[6]} :: {r[7]}")
    else:
        lines.append("\nNo earnings-call moonshot pipeline data yet for this ticker.")
    return "\n".join(lines)


class ChatStore:
    """One resumable claude CLI session per ticker. The CLI keeps the actual conversation
    history server-side (disk); this store only remembers which session ID belongs to
    which ticker so later turns can --resume it."""

    def __init__(self):
        self.sessions: dict[str, str] = {}
        self.lock = threading.Lock()

    def ask(self, ticker: str, question: str, context_fn) -> dict:
        ticker = ticker.upper()
        with self.lock:
            session_id = self.sessions.get(ticker)
        cmd = [
            "claude", "-p",
            (context_fn(ticker) + "\n\nQuestion: " + question) if session_id is None else question,
            "--output-format", "json",
            "--restricted", "--disallowedTools", "Bash", "Edit", "Write", "NotebookEdit",
            "--permission-prompts", "none",
        ]
        if session_id is None:
            session_id = str(uuid.uuid4())
            cmd += ["--session-id", session_id, "--system-prompt", CHAT_SYSTEM_PROMPT]
        else:
            cmd += ["--resume", session_id]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=CLAUDE_CHAT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "claude CLI timed out"}
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip() or result.stdout.strip() or "claude CLI failed"}
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"ok": False, "error": "could not parse claude CLI output"}
        if payload.get("is_error"):
            return {"ok": False, "error": payload.get("result") or "claude CLI reported an error"}
        with self.lock:
            self.sessions[ticker] = payload.get("session_id", session_id)
        return {"ok": True, "answer": payload.get("result", "")}

    def reset(self, ticker: str) -> None:
        with self.lock:
            self.sessions.pop(ticker.upper(), None)

# Turn a recent line of run_moonshot_pipeline.py's log into a short "what it's doing right now"
# phrase, checked in order against each line scanning from the newest backward. Patterns earlier
# in the list win on a given line; the first line (from the end) that matches anything wins overall.
_PHASE_PATTERNS: list[tuple[re.Pattern, callable]] = [
    (re.compile(r"^\[([\w.]+)\]\s+\d+\s+quarterly transcripts listed"), lambda m: "listing transcripts"),
    (re.compile(r"^\[[\w.]+\s+(\w+)\]\s+->\s+data/earnings_calls/"), lambda m: f"downloading transcript {m.group(1)}"),
    (re.compile(r"^\[[\d-]+\]\s+(\w+)\s+->\s+data/press_releases/"), lambda m: f"downloading press release {m.group(1)}"),
    (re.compile(r"^\[(\w+)\]\s+extracting ledger"), lambda m: f"extracting ledger {m.group(1)}"),
    (re.compile(r"^\[(\w+)\]\s+extracting press-release figures"), lambda m: f"reading press release {m.group(1)}"),
    (re.compile(r"^\[(\w+)\]\s+analyzing \("), lambda m: f"analysing {m.group(1)}"),
    (re.compile(r"^\[\w+_(\w+):(ambition|milestones|runway):\w+\]"), lambda m: f"writing {m.group(2)} {m.group(1)}"),
    (re.compile(r"^\d+ complete calls found"), lambda m: "scoring transcripts"),
    (re.compile(r"^\[(\w+)\]\s+ambition=\d"), lambda m: f"scoring {m.group(1)}"),
    (re.compile(r"^\[(\w+)\]\s+\d+ of \d+ quarters have a press release"), lambda m: "fetching press releases"),
    (re.compile(r"All steps complete"), lambda m: "wrapping up"),
]


def derive_phase(log_tail: str) -> str:
    """Scan the log tail from the newest line backward for the most recent recognizable step."""
    lines = log_tail.splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        for pattern, fmt in _PHASE_PATTERNS:
            m = pattern.search(line)
            if m:
                return fmt(m)
    return "starting"


class Job:
    def __init__(self, ticker: str, backend: str, proc: subprocess.Popen, log_path: Path):
        self.ticker = ticker
        self.backend = backend
        self.proc = proc
        self.log_path = log_path
        self.started = time.time()
        self.ended: float | None = None
        self.state = "running"  # running | done | failed | stopped

    def poll(self) -> None:
        if self.state != "running":
            return
        rc = self.proc.poll()
        if rc is None:
            return
        self.ended = time.time()
        self.state = "done" if rc == 0 else "failed"

    def log_tail(self) -> str:
        try:
            data = self.log_path.read_text(errors="replace")
        except OSError:
            return ""
        return data[-LOG_TAIL_CHARS:]

    def to_json(self) -> dict:
        self.poll()
        tail = self.log_tail()
        return {
            "ticker": self.ticker,
            "backend": self.backend,
            "state": self.state,
            "started": self.started,
            "ended": self.ended,
            "elapsed": (self.ended or time.time()) - self.started,
            "phase": derive_phase(tail) if self.state == "running" else self.state,
            "log_tail": tail,
        }


class JobStore:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def start(self, ticker: str, backend: str) -> dict:
        ticker = ticker.upper()
        with self.lock:
            existing = self.jobs.get(ticker)
            if existing is not None:
                existing.poll()
                if existing.state == "running":
                    return {"ok": False, "error": f"{ticker} is already running ({existing.backend})"}
            if backend not in BACKEND_ARGS:
                return {"ok": False, "error": f"unknown backend {backend!r}"}
            log_path = self.log_dir / f"{ticker}_{backend}_{int(time.time())}.log"
            cmd = [str(PY), str(PIPELINE), ticker, *BACKEND_ARGS[backend]]
            log_fh = log_path.open("w")
            proc = subprocess.Popen(
                cmd, cwd=REPO_ROOT, stdout=log_fh, stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group, so Stop can kill the whole tree
            )
            self.jobs[ticker] = Job(ticker, backend, proc, log_path)
            return {"ok": True, "ticker": ticker, "backend": backend}

    def stop(self, ticker: str) -> dict:
        ticker = ticker.upper()
        with self.lock:
            job = self.jobs.get(ticker)
            if job is None:
                return {"ok": False, "error": f"no job recorded for {ticker}"}
            job.poll()
            if job.state != "running":
                return {"ok": False, "error": f"{ticker} is not running (state={job.state})"}
            try:
                os.killpg(os.getpgid(job.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            job.state = "stopped"
            job.ended = time.time()
            return {"ok": True, "ticker": ticker}

    def status(self) -> dict:
        with self.lock:
            return {t: j.to_json() for t, j in self.jobs.items()}


def make_handler(store: JobStore, report_html: bytes, moonshot_dir: Path, chat: ChatStore, context_fn):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter default logging
            sys.stderr.write(f"[server] {self.address_string()} {fmt % args}\n")

        def _json(self, payload: dict, code: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(report_html)))
                self.end_headers()
                self.wfile.write(report_html)
            elif path == "/api/status":
                self._json(store.status())
            elif path.startswith("/api/scores/"):
                ticker = path.rsplit("/", 1)[-1].upper()
                data = _moonshot_payload([ticker], moonshot_dir)
                self._json({"ticker": ticker, "rows": data.get(ticker, [])})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                body = {}
            ticker = (body.get("ticker") or "").strip()
            if path == "/api/start":
                backend = (body.get("backend") or "nemo").strip()
                if not ticker:
                    self._json({"ok": False, "error": "ticker required"}, 400)
                    return
                self._json(store.start(ticker, backend))
            elif path == "/api/stop":
                if not ticker:
                    self._json({"ok": False, "error": "ticker required"}, 400)
                    return
                self._json(store.stop(ticker))
            elif path == "/api/chat":
                question = (body.get("question") or "").strip()
                if not ticker or not question:
                    self._json({"ok": False, "error": "ticker and question required"}, 400)
                    return
                self._json(chat.ask(ticker, question, context_fn))
            elif path == "/api/chat/reset":
                if not ticker:
                    self._json({"ok": False, "error": "ticker required"}, 400)
                    return
                chat.reset(ticker)
                self._json({"ok": True})
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def run_server(
    crossings_csv: Path | None, failures_csv: Path | None, data_dir: Path,
    crossings_scores: Path | None, failures_scores: Path | None,
    moonshot_dir: Path, host: str, port: int, open_browser: bool = True, rebuild: bool = False,
) -> None:
    fp = _fingerprint(crossings_csv, failures_csv, crossings_scores, failures_scores, moonshot_dir)
    cached = None
    if not rebuild and CACHE_HTML.exists() and CACHE_MANIFEST.exists():
        try:
            if json.loads(CACHE_MANIFEST.read_text()).get("fingerprint") == fp:
                cached = CACHE_HTML.read_text()
        except (OSError, json.JSONDecodeError):
            cached = None

    if cached is not None:
        print("Using cached report build (nothing changed since last build; pass --rebuild to force)",
              file=sys.stderr)
        report_html = cached
    else:
        out = Path("/tmp") / "inflections_report_live.html"
        build_inflections_report(crossings_csv, failures_csv, data_dir, out,
                                  crossings_scores, failures_scores, moonshot_dir)
        report_html = out.read_text()

        # build_inflections_report always emits `const DATA = {...};\nconst LISTS`. Turn that into
        # `const DATA = Object.assign({live:true}, {...});` so the template can tell a live serve
        # apart from a static export (and the buttons stay hidden on the plain exported file).
        marker = "const DATA = "
        start = report_html.index(marker) + len(marker)
        end = report_html.index(";\nconst LISTS", start)
        report_html = (
            report_html[:start] + "Object.assign({live:true}, " + report_html[start:end] + ")" + report_html[end:]
        )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_HTML.write_text(report_html)
        CACHE_MANIFEST.write_text(json.dumps({"fingerprint": fp}))

    store = JobStore(REPO_ROOT / "logs" / "screener_live")
    chat = ChatStore()
    context_fn = lambda t: _build_ticker_context(  # noqa: E731
        t, crossings_csv, failures_csv, crossings_scores, failures_scores, moonshot_dir
    )
    handler = make_handler(store, report_html.encode(), moonshot_dir, chat, context_fn)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"Serving live inflections report at {url}", file=sys.stderr)
    print(f"Pipeline: {PIPELINE}", file=sys.stderr)
    print(f"Python:   {PY}", file=sys.stderr)
    print(f"Moonshot scores read from: {moonshot_dir}", file=sys.stderr)
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
