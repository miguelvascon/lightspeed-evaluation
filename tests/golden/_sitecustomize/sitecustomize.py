"""Test-only startup hook for golden subprocesses (SPEC-01 §5.1 E).

Loaded because ``conftest.py`` puts this directory on ``PYTHONPATH``. The C7
import-time check runs without it. Installed only when ``LS_GOLDEN_NP_SEED``
is set and this module is the process ``sitecustomize``, so importing the file
during lint does not patch the linter.

The hook pins ``as_completed`` to submission order, seeds numpy, wraps
``Figure.savefig`` to dump a plotted-data fingerprint, blocks non-loopback
connects, and rejects ``oc`` / ``kubectl`` binaries that are not under the
allowlisted fake directory ``tests/golden/harness/fake_oc`` (override with
``LS_GOLDEN_FAKE_BIN``). Engineer 2 owns the scripts in that directory.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

_FAKE_BIN = Path(__file__).resolve().parents[1] / "harness" / "fake_oc"
_GUARDED_BINARIES = frozenset({"oc", "kubectl"})
_SPAWN = [subprocess.Popen]


def _pin_as_completed() -> None:
    """Yield futures in the order they were passed to ``as_completed``."""
    import concurrent.futures

    def as_completed(fs: Any, timeout: float | None = None) -> Any:
        futures = list(fs)
        if timeout is not None:
            _done, pending = concurrent.futures.wait(futures, timeout=timeout)
            if pending:
                raise concurrent.futures.TimeoutError()
        yield from futures

    marked: Any = as_completed
    marked.ls_golden_order_pin = True
    futures_mod: Any = concurrent.futures
    futures_mod.as_completed = as_completed


def _loopback(address: object) -> bool:
    if not isinstance(address, tuple) or not address:
        return True
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    return host in {"localhost", "::1"} or host.startswith("127.")


def _guard_sockets() -> None:
    """Fail connects that leave the loopback interface."""
    import socket

    def _wrap(real: Any) -> Any:
        def connect(self: Any, address: object) -> Any:
            if not _loopback(address):
                raise OSError("golden egress blocked")
            return real(self, address)

        return connect

    sock: Any = socket.socket
    sock.connect = _wrap(sock.connect)
    sock.connect_ex = _wrap(sock.connect_ex)


def _allowlisted(executable: Path) -> bool:
    root = Path(os.environ.get("LS_GOLDEN_FAKE_BIN", str(_FAKE_BIN))).resolve()
    try:
        resolved = executable.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def _reject_real_binary(args: object) -> None:
    if isinstance(args, (str, bytes)) or not isinstance(args, (list, tuple)):
        return
    if not args:
        return
    raw = args[0]
    if not isinstance(raw, (str, bytes)):
        return
    text = os.fsdecode(raw)
    name = Path(text).name
    if name not in _GUARDED_BINARIES:
        return
    candidate = Path(text)
    if not candidate.is_absolute():
        import shutil

        found = shutil.which(name)
        if found is None:
            raise OSError(f"golden isolation: {name} is not the allowlisted fake")
        candidate = Path(found)
    if not _allowlisted(candidate):
        raise OSError(f"golden isolation: {name} is not the allowlisted fake")


def _guard_binaries() -> None:
    """Allow ``oc`` and ``kubectl`` only from the golden fake-bin directory."""

    def popen(args: Any, *pargs: Any, **kwargs: Any) -> Any:
        _reject_real_binary(args)
        return _SPAWN[0](args, *pargs, **kwargs)

    module: Any = subprocess
    module.Popen = popen


def _rcparams(pyplot: Any) -> dict[str, str]:
    return {str(key): str(pyplot.rcParams[key]) for key in sorted(pyplot.rcParams)}


def _labels(axis: Any) -> list[str]:
    return [tick.get_text() for tick in axis.get_ticklabels()]


def _heatmap(
    collection: Any, xticks: list[str], yticks: list[str]
) -> list[dict[str, Any]]:
    import numpy as np

    values = np.asarray(collection.get_array(), dtype=float)
    if (
        values.ndim == 1
        and xticks
        and yticks
        and values.size == len(xticks) * len(yticks)
    ):
        values = values.reshape(len(yticks), len(xticks))
    if values.ndim != 2:
        return []
    rows: list[dict[str, Any]] = []
    for row_index, row_name in enumerate(yticks[: values.shape[0]]):
        for col_index, col_name in enumerate(xticks[: values.shape[1]]):
            rows.append(
                {
                    "key": f"{row_name}|{col_name}",
                    "value": float(values[row_index, col_index]),
                }
            )
    return rows


def _axis(ax: Any) -> dict[str, Any]:
    xticks = _labels(ax.xaxis)
    yticks = _labels(ax.yaxis)
    bars: list[dict[str, Any]] = []
    boxes: list[dict[str, Any]] = []
    wedges: list[dict[str, Any]] = []
    patches: list[dict[str, Any]] = []
    for patch in ax.patches:
        kind = type(patch).__name__
        if kind == "Wedge":
            label = (
                xticks[len(wedges)] if len(wedges) < len(xticks) else str(len(wedges))
            )
            span = float(patch.theta2) - float(patch.theta1)
            wedges.append({"key": label, "span": span})
        elif kind == "PathPatch":
            label = yticks[len(boxes)] if len(boxes) < len(yticks) else str(len(boxes))
            boxes.append({"key": label})
        elif kind == "Rectangle":
            label = xticks[len(bars)] if len(bars) < len(xticks) else str(len(bars))
            bars.append({"key": label, "height": float(patch.get_height())})
        else:
            patches.append({"type": kind})
    heatmap: list[dict[str, Any]] = []
    collections: list[dict[str, Any]] = []
    for collection in ax.collections:
        if type(collection).__name__ == "QuadMesh":
            heatmap.extend(_heatmap(collection, xticks, yticks))
        else:
            collections.append({"type": type(collection).__name__})
    legend = ax.get_legend()
    legend_text = (
        [] if legend is None else [text.get_text() for text in legend.get_texts()]
    )
    return {
        "title": ax.get_title(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xlim": [float(value) for value in ax.get_xlim()],
        "ylim": [float(value) for value in ax.get_ylim()],
        "xticks": xticks,
        "yticks": yticks,
        "legend": legend_text,
        "bars": bars,
        "boxes": boxes,
        "wedges": wedges,
        "heatmap": heatmap,
        "lines": [list(map(float, line.get_ydata())) for line in ax.lines],
        "texts": [text.get_text() for text in ax.texts],
        "patches": patches,
        "collections": collections,
    }


def _install_savefig() -> None:
    """Dump size, dpi, bbox, rcParams, and per-axes plotted data beside the PNG."""
    import json

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    original = Figure.savefig

    def savefig(self: Figure, *args: Any, **kwargs: Any) -> Any:
        self.canvas.draw()
        fingerprint = {
            "size": [float(value) for value in self.get_size_inches()],
            "dpi": kwargs.get("dpi", self.dpi),
            "bbox": kwargs.get("bbox_inches"),
            "rcparams": _rcparams(plt),
            "axes": [_axis(ax) for ax in self.axes],
        }
        target = args[0] if args else kwargs.get("fname")
        if isinstance(target, (str, os.PathLike)):
            path = os.fspath(target) + ".plotted.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(fingerprint, handle, sort_keys=True, default=str)
        return original(self, *args, **kwargs)

    marked: Any = savefig
    marked.ls_golden_savefig = True
    figure_cls: Any = Figure
    figure_cls.savefig = savefig


def _install() -> None:
    import numpy as np

    _pin_as_completed()
    _install_savefig()
    _guard_sockets()
    _guard_binaries()
    np.random.seed(int(os.environ["LS_GOLDEN_NP_SEED"]))


if __name__ == "sitecustomize" and os.environ.get("LS_GOLDEN_NP_SEED") is not None:
    _install()
