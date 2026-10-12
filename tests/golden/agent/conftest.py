"""Fixtures for the HTTP-agent goldens (WP0b).

``golden_env_extra`` and ``golden_pythonpath_extra`` are the WP0a per-directory
hooks. case.yaml ``env`` is the subprocess caller map (applied after them).
``pytest_marks`` supplies registered marks such as ``slow``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from tests.golden.harness._http_server import Router, prefix, serve
from tests.golden.harness.fake_lsc import FakeLsc
from tests.golden.harness.judge_server import JudgeReplay, load_cassette

HERE = Path(__file__).resolve().parent
_ROOT_CONFTEST = HERE.parent / "conftest.py"
ROOT_SKIP_REASON = (
    "h14 (lsc L19 chmod failure) skipped: running as root (euid 0); root can "
    "chmod any file, so the load-time chmod failure cannot be provoked"
)
_NOEXEC = ("/proc/version", "/etc/passwd", "/etc/hosts")


@dataclass
class CaseRun:
    """One CLI run of a copied agent fixture."""

    case: dict[str, Any]
    workdir: Path
    proc: subprocess.CompletedProcess[str]
    lsc: FakeLsc | None
    judge: JudgeReplay | None
    noexec_target: Path | None = None


@pytest.fixture
def golden_env_extra() -> dict[str, str]:
    """Extra env for every HTTP case. Per-case keys live in case.yaml ``env``."""
    return {}


@pytest.fixture
def golden_pythonpath_extra() -> list[str]:
    """Extra ``PYTHONPATH`` entries. HTTP cases do not prepend any."""
    return []


def load_case(fixture_dir: Path) -> dict[str, Any]:
    """Parse ``case.yaml`` for one fixture directory."""
    loaded = yaml.safe_load((fixture_dir / "case.yaml").read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{fixture_dir} case.yaml must be a mapping")
    return loaded


def discover(kind: str) -> list[Path]:
    """Return fixture directories under ``fixtures/<kind>`` that have case.yaml."""
    root = HERE / "fixtures" / kind
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if (path / "case.yaml").is_file())


def case_params(kind: str) -> list[Any]:
    """One pytest param per fixture, including marks from ``pytest_marks``."""
    params = []
    for folder in discover(kind):
        marks = [
            getattr(pytest.mark, name)
            for name in load_case(folder).get("pytest_marks", [])
        ]
        params.append(pytest.param(folder, id=folder.name, marks=marks))
    return params


def _publish_openai_base_url(url: str) -> None:
    for module in list(sys.modules.values()):
        if getattr(module, "__file__", None) != str(_ROOT_CONFTEST):
            continue
        hook = cast(
            Callable[[str], None] | None,
            getattr(module, "publish_openai_base_url", None),
        )
        if hook is not None:
            hook(url)


def _copy_fixture(src: Path, dst: Path, subs: dict[str, str]) -> dict[str, Any]:
    shutil.copytree(src, dst)
    for name in ("system.yaml", "eval_data.yaml", "case.yaml"):
        path = dst / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for key, value in subs.items():
            text = text.replace(key, value)
        path.write_text(text, encoding="utf-8")
    case = load_case(dst)
    for rel, mode in (case.get("chmod") or {}).items():
        os.chmod(dst / str(rel), int(str(mode), 8))
    return case


def _noexec_target() -> Path | None:
    for cand in _NOEXEC:
        path = Path(cand)
        if not path.is_file() or os.access(path, os.X_OK):
            continue
        try:
            os.chmod(path, path.stat().st_mode & 0o7777)
        except PermissionError:
            return path.resolve()
    return None


def _link_noexec(work: Path, case: dict[str, Any]) -> Path | None:
    links = case.get("noexec_link") or {}
    if not links:
        return None
    if os.geteuid() == 0:
        pytest.skip(ROOT_SKIP_REASON)
    target = _noexec_target()
    if target is None:
        pytest.skip(
            "h14 (lsc L19 chmod failure) skipped: no file this user cannot chmod"
        )
    for rel in links:
        dest = work / str(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(target)
    return target


def _mount(
    fixture_dir: Path, log: Path
) -> tuple[Router, FakeLsc | None, JudgeReplay | None]:
    case = load_case(fixture_dir)
    fake = None
    judge = None
    mounted = []
    script = fixture_dir / "lsc_script.json"
    if script.is_file():
        fake = FakeLsc.from_files(script, log)
        mounted.append(prefix("/lsc", fake.http_handler()))
    cassette = case.get("judge")
    if cassette:
        judge = JudgeReplay(load_cassette(fixture_dir / str(cassette)))
        mounted.append(prefix("/v1", judge.handle))
    return Router(*mounted), fake, judge


def _cli(work: Path, caller: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(Path(sys.executable).with_name("lightspeed-eval")),
            "--system-config",
            str(work / "system.yaml"),
            "--eval-data",
            str(work / "eval_data.yaml"),
        ],
        cwd=work,
        env=caller or None,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.fixture
def run_agent_case(tmp_path: Path) -> Callable[[Path], CaseRun]:
    """Return ``run(fixture_dir)`` that executes one case against loopback."""

    def _run(fixture_dir: Path) -> CaseRun:
        router, fake, judge = _mount(fixture_dir, tmp_path / "lsc_requests.jsonl")
        with serve(router) as base:
            if judge is not None:
                _publish_openai_base_url(f"{base}/v1")
            work = tmp_path / "case"
            case = _copy_fixture(
                fixture_dir,
                work,
                {"@LSC_BASE@": f"{base}/lsc", "@OUT@": str(work)},
            )
            noexec = _link_noexec(work, case)
            proc = _cli(
                work,
                {
                    str(key): str(value)
                    for key, value in (case.get("env") or {}).items()
                },
            )
        return CaseRun(case, work, proc, fake, judge, noexec)

    return _run
