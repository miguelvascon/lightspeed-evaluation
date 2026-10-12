#!/usr/bin/env python3
"""Run a command in its own PTY and save a transcript plus the child pid.

Usage:
  pty_run.py --pid-file PATH --transcript PATH --cwd DIR -- CMD [ARGS...]

Exits with the command's status. On timeout (300s) kills the process group
and exits 124. Strips carriage returns so the transcript can be searched.
"""

from __future__ import annotations

import argparse
import os
import pty
import select
import subprocess
import sys
import time

TIMEOUT_SECONDS = 300


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        parser.error("command is required after --")
    return args


def main() -> int:
    """Allocate a PTY, run the command, and write the transcript."""
    args = _parse_args()
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        args.cmd,
        cwd=args.cwd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave_fd)
    with open(args.pid_file, "w", encoding="utf-8") as pid_file:
        pid_file.write(f"{proc.pid}\n")

    timed_out = False
    with open(args.transcript, "wb") as transcript:
        deadline = time.monotonic() + TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            readable, _, _ = select.select([master_fd], [], [], min(remaining, 0.5))
            if master_fd in readable:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                transcript.write(chunk.replace(b"\r", b""))
                sys.stdout.buffer.write(chunk.replace(b"\r", b""))
                sys.stdout.buffer.flush()
            if proc.poll() is not None and not readable:
                # Drain anything left after exit.
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0)
                    if not ready:
                        break
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        break
                    transcript.write(chunk.replace(b"\r", b""))
                    sys.stdout.buffer.write(chunk.replace(b"\r", b""))
                    sys.stdout.buffer.flush()
                break

    os.close(master_fd)
    if timed_out:
        try:
            os.killpg(proc.pid, 15)
        except OSError:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except OSError:
                proc.kill()
            proc.wait()
        print(
            f"pty_run: timed out after {TIMEOUT_SECONDS}s",
            file=sys.stderr,
        )
        return 124

    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
