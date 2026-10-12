"""Shared pieces for the DS cassette tools (author_cassettes.py, record_cassettes.py).

- ``JudgeProxy``: an OpenAI-compatible server on 127.0.0.1 with four modes.
  ``fake`` answers from a responder function, ``record`` forwards to an upstream,
  ``replay`` serves a cassette strictly (miss → HTTP 599), and ``off`` refuses.
- ``SITECUSTOMIZE``: the golden startup hook (SPEC-01 §5.1 E), injected via
  PYTHONPATH. It has a loopback-only socket guard, the ``as_completed``
  submission-order pin, and ``numpy.random.seed(LS_GOLDEN_NP_SEED)``.
- CLI runner, output capture and normalization (§5.1 A–D), cassette I/O and
  the secret scan.

E3 may fold ``JudgeProxy`` into ``harness/judge_server.py``. The cassette line
format and replay key are the contract: sha256(b"POST " + path + b"\\n" + body)
plus a per-key ordinal.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import http.server
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{8,}"),
]
API_KEY_FIELD = re.compile(r'("api_key"\s*:\s*)"[^"]*"')
ENV_DROP_PREFIXES = (
    "OPENAI_", "AZURE_", "ANTHROPIC_", "GEMINI_", "GOOGLE_", "WATSONX_",
    "HOSTED_VLLM_", "LITELLM_", "LANGFUSE_", "MLFLOW_",
)
ENV_DROP_EXACT = {"API_KEY", "OLLAMA_HOST", "KUBECONFIG", "DEEPEVAL_API_KEY", "SSL_CERT_FILE"}
CLI_PLACEHOLDER_KEY = "golden-fake"
TIMING_FIELDS = {"execution_time", "evaluation_latency", "agent_latency",
                 "time_to_first_token", "streaming_duration", "tokens_per_second"}

Responder = Callable[[str, dict], "tuple[int, dict]"]


def scrub_text(text: str, secret: Optional[str]) -> tuple[str, bool]:
    out = API_KEY_FIELD.sub(r'\1"<REDACTED>"', text)
    if secret:
        out = out.replace(secret, "<REDACTED>")
    for pat in KEY_PATTERNS:
        out = pat.sub("<REDACTED>", out)
    return out, out != text


def exchange_key(path: str, body: str) -> str:
    return hashlib.sha256(f"POST {path}\n".encode() + body.encode()).hexdigest()


# --------------------------------------------------------------------- proxy
class JudgeProxy:
    def __init__(self, responder: Optional[Responder] = None) -> None:
        self.responder = responder
        self.mode = "fake" if responder else "off"
        self.upstream: Optional[str] = None
        self._secret: Optional[str] = None
        self.cap: Optional[int] = None
        self.forwarded = 0
        self.abort_reason: Optional[str] = None
        self.lock = threading.Lock()
        self.exchanges: list[dict] = []
        self.ordinals: dict[tuple[str, str], int] = {}
        self.replay_index: dict[str, list[dict]] = {}
        self.replay_used: dict[str, int] = {}
        self.misses: list[dict] = []
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("content-length", 0)))
                status, payload = proxy.handle(self.path, raw)
                out = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def do_GET(self) -> None:  # noqa: N802
                self.send_response(404)
                self.send_header("content-length", "0")
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                return

        class Server(http.server.ThreadingHTTPServer):
            def handle_error(self, request, client_address) -> None:  # client hung up: not our error
                return

        self.server = Server(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def set_fake(self, responder: Optional[Responder] = None) -> None:
        if responder:
            self.responder = responder
        self.mode, self.upstream, self._secret, self.cap = "fake", None, None, None

    def set_record(self, upstream: str, secret: str, cap: int) -> None:
        self.mode, self.upstream, self._secret, self.cap = "record", upstream.rstrip("/"), secret, cap

    def set_replay(self, entries: list[dict]) -> None:
        self.mode, self.upstream, self._secret, self.cap = "replay", None, None, None
        self.replay_index, self.replay_used, self.misses = {}, {}, []
        for e in sorted(entries, key=lambda e: (e["key"], e["ordinal"])):
            self.replay_index.setdefault(e["key"], []).append(e)

    def reset(self) -> None:
        with self.lock:
            self.exchanges, self.ordinals = [], {}
            self.forwarded, self.abort_reason = 0, None

    def base_url(self, tag: str) -> str:
        return f"http://127.0.0.1:{self.port}/t/{tag}/v1"

    def handle(self, raw_path: str, raw: bytes) -> tuple[int, dict]:
        m = re.match(r"^/t/([^/]+)(/v1/.*)$", raw_path)
        tag, path = (m.group(1), m.group(2)) if m else ("default", raw_path)
        body_text = raw.decode("utf-8", errors="replace")
        scrubbed, changed = scrub_text(body_text, self._secret)
        key = exchange_key(path, scrubbed)
        with self.lock:
            if self.abort_reason:
                return 400, {"error": {"message": "aborted"}}
            ordinal = self.ordinals.get((tag, key), 0)
            self.ordinals[(tag, key)] = ordinal + 1
        try:
            body = json.loads(body_text)
        except ValueError:
            body = {}

        if self.mode == "replay":
            with self.lock:
                used = self.replay_used.get(key, 0)
                cands = self.replay_index.get(key, [])
                if used >= len(cands):
                    self.misses.append({"tag": tag, "path": path, "key": key, "body": scrubbed[:600]})
                    return 599, {"error": {"message": f"golden replay miss {key[:12]}"}}
                self.replay_used[key] = used + 1
                entry = cands[used]
            self._log(tag, path, key, ordinal, scrubbed, changed, entry["response"], "replay")
            return entry["response"]["status"], entry["response"]["body"]
        if self.mode == "off":
            return 503, {"error": {"message": "golden proxy is off"}}

        is_embedding = path.endswith("/embeddings")
        if self.mode == "fake" or is_embedding:
            assert self.responder is not None
            if body.get("stream"):
                with self.lock:
                    self.abort_reason = "streaming judge request is not supported"
                return 400, {"error": {"message": self.abort_reason}}
            status, resp = self.responder(path, body)
            source = "handwritten-embedding" if is_embedding else "handwritten"
        else:  # record
            with self.lock:
                if self.cap is not None and self.forwarded >= self.cap:
                    self.abort_reason = f"hard cap of {self.cap} upstream requests reached"
                    return 400, {"error": {"message": self.abort_reason}}
                self.forwarded += 1
            status, resp = self._forward(path, raw)
            source = "recorded"
            if not 200 <= status < 300:
                with self.lock:
                    self.abort_reason = f"upstream returned HTTP {status}"
                return 400, {"error": {"message": "recording aborted (upstream error)"}}
        self._log(tag, path, key, ordinal, scrubbed, changed, {"status": status, "body": resp}, source)
        return status, resp

    def _forward(self, path: str, raw: bytes) -> tuple[int, dict]:
        assert self.upstream and self._secret
        req = urllib.request.Request(self.upstream + path[len("/v1"):], data=raw, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + self._secret})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:  # nosec B310 - https upstream only
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode())
            except ValueError:
                return e.code, {"error": {"message": "non-JSON upstream error"}}
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return 598, {"error": {"message": f"upstream unreachable: {type(e).__name__}"}}

    def _log(self, tag, path, key, ordinal, body, scrubbed, response, source) -> None:
        with self.lock:
            self.exchanges.append({"tag": tag, "path": path, "key": key, "ordinal": ordinal,
                                   "body": body, "scrubbed": scrubbed,
                                   "response": response, "source": source})

    def unused(self) -> list[str]:
        return [k for k, v in self.replay_index.items() if self.replay_used.get(k, 0) < len(v)]


# ---------------------------------------------------------------- golden env
SITECUSTOMIZE = '''
"""Golden startup hook (test-only; SPEC-01 §5.1 E). Not used by the C7 import-time golden."""
import os, socket
_LOG = os.environ.get("LS_GOLDEN_NET_LOG")
_orig_connect = socket.socket.connect
_orig_gai = socket.getaddrinfo
def _is_local(host):
    return host in ("127.0.0.1", "::1", "localhost") or str(host).startswith("127.")
def _note(what):
    if _LOG:
        with open(_LOG, "a") as f:
            f.write(what + "\\n")
def _guard_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(host):
        _note("BLOCKED connect %r" % (host,))
        raise OSError("golden guard: non-loopback connection blocked")
    return _orig_connect(self, address)
def _guard_gai(host, *a, **k):
    if host is not None and not _is_local(host):
        _note("BLOCKED resolve %r" % (host,))
        raise socket.gaierror("golden guard: non-loopback resolution blocked")
    return _orig_gai(host, *a, **k)
socket.socket.connect = _guard_connect
socket.getaddrinfo = _guard_gai

if os.environ.get("LS_GOLDEN_PIN") == "1":
    import concurrent.futures as _cf
    def _as_completed_in_submission_order(fs, timeout=None):
        for f in list(fs):
            _cf.wait([f], timeout=timeout)
            yield f
    _cf.as_completed = _as_completed_in_submission_order
if os.environ.get("LS_GOLDEN_NP_SEED") is not None:
    import numpy.random as _npr
    _npr.seed(int(os.environ["LS_GOLDEN_NP_SEED"]))
'''


def write_sitecustomize(guard_dir: Path) -> None:
    guard_dir.mkdir(parents=True, exist_ok=True)
    (guard_dir / "sitecustomize.py").write_text(SITECUSTOMIZE)


def cli_env(base_url: str, guard_dir: str, net_log: str, mpl_dir: str,
            unset: tuple[str, ...] = (), extra: Optional[dict] = None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(ENV_DROP_PREFIXES) and k not in ENV_DROP_EXACT}
    env.update({
        "OPENAI_API_KEY": CLI_PLACEHOLDER_KEY,
        "OPENAI_BASE_URL": base_url,
        "PYTHONHASHSEED": "0", "TZ": "UTC", "LC_ALL": "C.UTF-8", "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": mpl_dir, "LITELLM_LOCAL_MODEL_COST_MAP": "True", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
        "PYTHONPATH": guard_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "LS_GOLDEN_NET_LOG": net_log, "LS_GOLDEN_PIN": "1", "LS_GOLDEN_NP_SEED": "0",
        "PYTHONUNBUFFERED": "1",
    })
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(k, None)
    env.update(extra or {})
    for k in unset:
        env.pop(k, None)
    return env


def run_cli(exe: list[str], cwd: Path, system: Path, data: Path, out_dir: Path, env: dict,
            proxy: Optional[JudgeProxy] = None, extra_args: tuple[str, ...] = (),
            timeout: int = 1800) -> int:
    cmd = exe + ["--system-config", str(system), "--eval-data", str(data),
                 "--output-dir", str(out_dir), *extra_args]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir.parent / f"{out_dir.name}.cli.log", "w") as logf:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            if (proxy is not None and proxy.abort_reason) or time.monotonic() > deadline:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return -9
            time.sleep(0.2)
    return proc.returncode


# ------------------------------------------------------------- normalization
_TS_NAME = re.compile(r"_\d{8}_\d{6}")
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+00:00|Z)?")
_ADDR = re.compile(r"\b0x[0-9a-f]{6,16}\b")  # object reprs in exception text (proven to vary)
_LOGLINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ")


def _norm_str(s: str, subs: dict[str, str]) -> str:
    for k, v in subs.items():
        s = s.replace(k, v)
    return _ADDR.sub("<ADDR>", _ISO.sub("<TS>", _TS_NAME.sub("_<TS>", s)))


def _norm_obj(o: Any, subs: dict[str, str], key: str = "") -> Any:
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k == "timestamp":
                out[k] = "<TS>"
            elif k in ("agent_latency_stats", "streaming"):
                out[k] = _norm_obj(v, subs, "__timing__")
            else:
                out[k] = _norm_obj(v, subs, k)
        return out
    if isinstance(o, list):
        return [_norm_obj(v, subs, key) for v in o]
    if isinstance(o, float) and (key in TIMING_FIELDS or key == "__timing__") and o != 0.0:
        return "<T>"
    if isinstance(o, str):
        return _norm_str(o, subs)
    return o


def capture(out_dir: Path, rc: int, subs: dict[str, str]) -> dict:
    """Everything the golden compares, normalized per SPEC-01 §5.1 (A, B, D)."""
    res: dict[str, Any] = {"exit_code": rc}
    files = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file())
    res["files"] = [_norm_str(f, subs) for f in files]
    csvs = glob.glob(str(out_dir / "*_detailed.csv"))
    rows: list[dict] = []
    if csvs:
        with open(csvs[0], newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({k: ("<T>" if k in TIMING_FIELDS and v not in ("", "0.0", "0", None)
                                 else _norm_str(v or "", subs)) for k, v in r.items()})
    res["csv_rows"] = rows
    sums = glob.glob(str(out_dir / "*_summary.json"))
    res["summary"] = _norm_obj(json.loads(Path(sums[0]).read_text()), subs) if sums else None
    txts = glob.glob(str(out_dir / "*_summary.txt"))
    if txts:
        lines = Path(txts[0]).read_text(encoding="utf-8").splitlines()
        res["summary_txt"] = [_norm_str(re.sub(r"\d+\.\d+s\b", "<T>s", l), subs)
                              if not l.startswith("Generated:") else "Generated: <TS>" for l in lines]
    pngs = sorted(out_dir.rglob("*.png"))
    res["png_sha256"] = {_norm_str(p.relative_to(out_dir).as_posix(), subs):
                         hashlib.sha256(p.read_bytes()).hexdigest() for p in pngs}
    log = out_dir.parent / f"{out_dir.name}.cli.log"
    console = []
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.rstrip()
            if (not line or _LOGLINE.match(line) or "LiteLLM" in line or "%|" in line
                    or line.startswith(("  File ", "    ")) or "Traceback" in line
                    or line.startswith("DEPRECATION")):
                continue
            console.append(_norm_str(line, subs))
    res["console"] = console
    return res


# ------------------------------------------------------------------ cassettes
def to_entries(exchanges: list[dict], tag: str) -> list[dict]:
    rows = []
    for e in sorted((x for x in exchanges if x["tag"] == tag), key=lambda x: (x["key"], x["ordinal"])):
        rows.append({"key": e["key"], "ordinal": e["ordinal"],
                     "request": {"method": "POST", "path": e["path"], "body": e["body"]},
                     "response": e["response"], "source": e["source"], "scrubbed": e["scrubbed"]})
    return rows


def per_key_conflicts(entries: list[dict]) -> list[str]:
    seen: dict[str, str] = {}
    bad = []
    for e in entries:
        r = json.dumps(e["response"], sort_keys=True)
        if e["key"] in seen and seen[e["key"]] != r:
            bad.append(e["key"])
        seen.setdefault(e["key"], r)
    return sorted(set(bad))


def secret_scan(path: Path, secret: Optional[str]) -> list[str]:
    hits = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if secret and secret in line:
            hits.append(f"line {i}: contains the OPENAI_API_KEY value")
        for pat in KEY_PATTERNS:
            if pat.search(line):
                hits.append(f"line {i}: matches {pat.pattern}")
        for m in API_KEY_FIELD.finditer(line):
            if '"<REDACTED>"' not in m.group(0) and not m.group(0).endswith('""'):
                hits.append(f"line {i}: unredacted api_key field")
        if re.search(r'(?i)"authorization"', line):
            hits.append(f"line {i}: contains an Authorization header")
    return hits


def write_cassette(entries: list[dict], dest: Path, secret: Optional[str]) -> list[str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n")
    hits = secret_scan(tmp, secret)
    if hits:
        tmp.unlink()
        return hits
    os.replace(tmp, dest)
    return []


def load_cassette(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
