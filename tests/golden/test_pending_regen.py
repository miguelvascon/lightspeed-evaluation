"""PENDING_REGEN parsing, skip, and the two-PR protocol."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.golden._compare import (
    PendingRegenError,
    apply_pending,
    assert_pending_empty,
    main,
    parse_pending_regen,
    pending_protocol_error,
    pending_review_warning,
    pending_summary,
    rate_fixture,
    reject_unknown_cases,
)
from tests.golden._regen import regen_guard_blocks
from tests.golden.conftest import known_case_ids, skip_pending_items

pytestmark = pytest.mark.golden

_LISTED = "ragas_paths RM-15\n"


def _item(case: str | None = None, **params: Any) -> Any:
    node = SimpleNamespace(case=case, added=[], callspec=SimpleNamespace(params=params))

    def get_closest_marker(name: str) -> Any:
        if name != "golden_case" or node.case is None:
            return None
        return SimpleNamespace(args=(node.case,))

    def add_marker(marker: Any) -> None:
        node.added.append(marker)

    node.get_closest_marker = get_closest_marker
    node.add_marker = add_marker
    return node


def test_unlisted_diff_still_fails() -> None:
    """A diff on a case that is not listed stays FAIL."""
    rating = rate_fixture({"score": 0.0}, {"score": 1.0}, compare_png=True)
    assert rating == "FAIL"
    pending = {"ragas_paths": "RM-15"}
    assert apply_pending("nlp_metrics", rating, pending) == "FAIL"
    assert apply_pending("ragas_paths", rating, pending) == "PENDING"
    assert apply_pending("nlp_metrics", rating, {}) == "FAIL"


def _git(repo: Path, *args: str) -> str:
    command = [
        "git",
        "-C",
        str(repo),
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        *args,
    ]
    return subprocess.check_output(command, text=True, timeout=30)


def test_listed_case_is_pending_and_skipped() -> None:
    """A listed case is PENDING and receives a skip marker. Others do not."""
    pending = parse_pending_regen(_LISTED)
    listed = _item("ragas_paths")
    other = _item("nlp_metrics")
    skip_pending_items([listed, other], pending)
    assert pending_summary(pending) == "PENDING: ragas_paths RM-15"
    assert len(listed.added) == 1
    assert listed.added[0].mark.kwargs["reason"] == "PENDING ragas_paths RM-15"
    assert other.added == []


def test_parametrized_case_skips_without_marker(tmp_path: Path) -> None:
    """A ``case_id`` param or a ``fixtures/`` directory skips with no marker."""
    pending = parse_pending_regen(_LISTED)
    by_param = _item(case_id="ragas_paths")
    other = _item(case_id="nlp_metrics")
    folder = tmp_path / "fixtures" / "ragas_paths"
    folder.mkdir(parents=True)
    by_dir = _item(fixture=folder)
    skip_pending_items([by_param, other, by_dir], pending)
    assert by_param.added[0].mark.kwargs["reason"] == "PENDING ragas_paths RM-15"
    assert by_dir.added[0].mark.kwargs["reason"] == "PENDING ragas_paths RM-15"
    assert other.added == []


def test_unknown_case_id_is_rejected(tmp_path: Path) -> None:
    """``not_a_case`` raises. A fixture directory with system.yaml is known."""
    case = tmp_path / "metrics" / "fixtures" / "ragas_paths"
    case.mkdir(parents=True)
    (case / "system.yaml").write_text("agents: {}\n")
    known = known_case_ids([], tmp_path)
    assert known == {"ragas_paths"}
    reject_unknown_cases({"ragas_paths": "RM-15"}, known)
    with pytest.raises(PendingRegenError, match="not_a_case"):
        reject_unknown_cases(parse_pending_regen("not_a_case RM-99\n"), known)


def test_src_addition_warns_for_ds_review() -> None:
    """A src/ PR that adds PENDING entries warns and names each one."""
    src = ["src/lightspeed_evaluation/api.py", "tests/golden/PENDING_REGEN"]
    warning = pending_review_warning("", _LISTED, src)
    assert warning is not None
    assert "ragas_paths RM-15" in warning
    assert "DS review" in warning
    assert pending_review_warning("", _LISTED, ["tests/golden/PENDING_REGEN"]) is None
    assert pending_review_warning(_LISTED, _LISTED, src) is None


def test_rename_out_of_src_blocks_both_guards(tmp_path: Path) -> None:
    """``--no-renames`` keeps ``src/a.py`` after ``git mv`` out of ``src/``."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x\n")
    golden = repo / "tests" / "golden"
    golden.mkdir(parents=True)
    (golden / "PENDING_REGEN").write_text(_LISTED)
    expected = golden / "metrics" / "expected" / "case"
    expected.mkdir(parents=True)
    (expected / "out.json").write_text("{}\n")
    _git(repo, "init", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "other").mkdir()
    _git(repo, "mv", "src/a.py", "other/a.py")
    (golden / "PENDING_REGEN").write_text("")
    (expected / "out.json").write_text('{"n": 1}\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "move")
    head = _git(repo, "rev-parse", "HEAD").strip()
    folded = _git(repo, "diff", "--name-only", f"{base}...{head}").split()
    opened = _git(
        repo, "diff", "--name-only", "--no-renames", f"{base}...{head}"
    ).split()
    two_dot = _git(repo, "diff", "--name-only", "--no-renames", base, head).split()
    assert "src/a.py" not in folded
    assert "src/a.py" in opened
    assert "src/a.py" in two_dot
    assert regen_guard_blocks(folded, []) is False
    assert regen_guard_blocks(opened, []) is True
    assert regen_guard_blocks(two_dot, []) is True
    assert pending_protocol_error(_LISTED, "", folded) is None
    assert pending_protocol_error(_LISTED, "", opened) is not None


def test_require_empty_blocks_a_release() -> None:
    """G0 rejects a listed case. The checked-in file passes ``--require-empty``."""
    with pytest.raises(PendingRegenError, match="ragas_paths RM-15"):
        assert_pending_empty(_LISTED)
    assert main(["--require-empty"]) == 0
    assert main([]) == 2


def test_require_empty_rejects_a_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Deleting ``PENDING_REGEN`` fails G0. An empty file in that path passes."""
    monkeypatch.chdir(tmp_path)
    assert main(["--require-empty"]) == 1
    assert "PENDING_REGEN is missing" in capsys.readouterr().err
    pending = tmp_path / "tests" / "golden"
    pending.mkdir(parents=True)
    (pending / "PENDING_REGEN").write_text("", encoding="utf-8")
    assert main(["--require-empty"]) == 0


def test_empty_file_is_a_noop() -> None:
    """The checked-in file lists nothing, so no case is skipped."""
    text = Path("tests/golden/PENDING_REGEN").read_text(encoding="utf-8")
    pending = parse_pending_regen(text)
    assert not pending
    assert pending_summary(pending) == "PENDING: none"
    node = _item("ragas_paths")
    skip_pending_items([node], pending)
    assert node.added == []
    assert pending_protocol_error("", "", ["src/lightspeed_evaluation/api.py"]) is None


@pytest.mark.parametrize(
    "text",
    [
        "ragas_paths",
        "ragas_paths rm-15",
        "ragas_paths RM-",
        "ragas_paths RM-0",
        "Ragas_paths RM-15",
        "ragas-paths RM-15",
        "ragas_paths RM-15 extra",
        "ragas_paths RM-15\nragas_paths RM-16\n",
    ],
)
def test_malformed_entries_are_rejected(text: str) -> None:
    """Reject a line that is not ``case_id RM-<number>``, and duplicates."""
    with pytest.raises(PendingRegenError):
        parse_pending_regen(text)


def test_protocol_requires_the_regen_pr_next() -> None:
    """A non-empty base blocks every PR except the one that empties the list."""
    base = _LISTED
    unrelated = ["docs/README.md"]
    assert pending_protocol_error(base, base, unrelated) is not None
    regen = [
        "tests/golden/PENDING_REGEN",
        "tests/golden/metrics/expected/ragas_paths/out.json",
    ]
    assert pending_protocol_error(base, "", regen) is None
    with_src = ["tests/golden/PENDING_REGEN", "src/lightspeed_evaluation/api.py"]
    assert pending_protocol_error(base, "", with_src) is not None
    assert pending_protocol_error(base, "nlp_metrics RM-16\n", regen) is not None
    assert pending_protocol_error("not a line\n", "", regen) is not None
