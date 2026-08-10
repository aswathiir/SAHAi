"""Execution of untrusted, model-generated code.

This is the only component in the system permitted to run code it did not
write. Isolation is layered, because any single layer can be bypassed:

  1. Container   — no network, read-only rootfs, non-root user, dropped
                   capabilities, seccomp. Configured in the Dockerfile and
                   compose file, not here.
  2. Process     — a fresh interpreter per test case, never the serving one.
  3. rlimits     — address space, CPU seconds, file descriptors, subprocess
                   count, and file size, applied in the child before exec.
  4. Wall clock  — hard timeout with process-group kill, so a child that forks
                   cannot outlive it.

Layer 1 is the one that actually contains a determined escape. The rest exist
so that ordinary runaway code fails fast and cheaply.
"""

from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from sahai_core import ExecutionLimits, ExecutionOutcome

# Wall-clock grace beyond the CPU limit, so we can distinguish a genuine
# timeout from a CPU-bound loop the rlimit already killed.
_WALL_GRACE_SECONDS = 2

# Signals that mean "the sandbox stopped it", not "the candidate crashed".
_RESOURCE_KILL_SIGNALS = frozenset(
    getattr(signal, name).value
    for name in ("SIGKILL", "SIGXCPU", "SIGXFSZ")
    if hasattr(signal, name)
)


@dataclass(frozen=True)
class RunOutcome:
    outcome: ExecutionOutcome
    stdout: str | None = None
    error: str | None = None


def _preexec(limits: ExecutionLimits):  # pragma: no cover - runs in the child
    """Applied inside the child between fork and exec.

    Anything raised here surfaces as an opaque SubprocessError, so every limit
    is applied defensively: not all of them exist on every platform (RLIMIT_AS
    is unreliable on macOS, RLIMIT_NPROC is absent on some libcs), and a missing
    limit must not stop the run. The container is the real boundary; these are
    fast-failure guards.

    The session is created by `start_new_session=True` on the Popen call, not
    here — calling os.setsid() as well fails with EPERM.
    """
    mem = limits.memory_mb * 1024 * 1024
    wanted = [
        ("RLIMIT_AS", mem),
        ("RLIMIT_DATA", mem),
        ("RLIMIT_CPU", limits.timeout_seconds),
        ("RLIMIT_NOFILE", 64),
        ("RLIMIT_FSIZE", 1024 * 1024),
        ("RLIMIT_NPROC", 64),
    ]

    def apply() -> None:
        for name, value in wanted:
            limit = getattr(resource, name, None)
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                continue

    return apply


def _build_script(code: str, call: str) -> str:
    """Wrap the candidate so its single return value is emitted as JSON.

    The result is printed to a dedicated sentinel line rather than bare stdout,
    so anything the candidate itself prints cannot be mistaken for the answer.
    """
    return (
        "import json, sys\n"
        f"{code}\n"
        "\n"
        "def __sahai_main():\n"
        f"    return {call}\n"
        "\n"
        "__sahai_result = __sahai_main()\n"
        "sys.stdout.write('\\n__SAHAI_RESULT__' + json.dumps(__sahai_result, default=str))\n"
    )


_SENTINEL = "__SAHAI_RESULT__"


def run_one(
    code: str,
    call: str,
    expected: object,
    limits: ExecutionLimits | None = None,
) -> RunOutcome:
    """Execute one candidate against one test case."""
    limits = limits or ExecutionLimits()
    script = _build_script(code, call)

    with tempfile.TemporaryDirectory(prefix="sahai-exec-") as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", script],
                capture_output=True,
                text=True,
                timeout=limits.timeout_seconds + _WALL_GRACE_SECONDS,
                cwd=workdir,
                preexec_fn=_preexec(limits),
                env={"PATH": "/usr/bin:/bin", "HOME": workdir, "PYTHONHASHSEED": "0"},
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return RunOutcome(ExecutionOutcome.TIMEOUT, error="wall-clock timeout")
        except (OSError, ValueError) as exc:
            return RunOutcome(ExecutionOutcome.ERROR, error=f"spawn failed: {exc}")

    # RLIMIT_CPU delivers SIGXCPU before escalating to SIGKILL, so checking only
    # for SIGKILL misreports a CPU-exhausted run as a generic error.
    if -proc.returncode in _RESOURCE_KILL_SIGNALS:
        return RunOutcome(
            ExecutionOutcome.TIMEOUT,
            error=f"killed by {signal.Signals(-proc.returncode).name} (resource limit)",
        )
    if proc.returncode != 0:
        return RunOutcome(
            ExecutionOutcome.ERROR,
            error=_truncate(proc.stderr, limits.max_output_bytes),
        )

    stdout = proc.stdout
    if _SENTINEL not in stdout:
        return RunOutcome(
            ExecutionOutcome.ERROR,
            stdout=_truncate(stdout, limits.max_output_bytes),
            error="candidate produced no result",
        )

    payload = stdout.rsplit(_SENTINEL, 1)[1]
    try:
        actual = json.loads(payload)
    except json.JSONDecodeError:
        return RunOutcome(
            ExecutionOutcome.ERROR,
            stdout=_truncate(payload, limits.max_output_bytes),
            error="result was not JSON-serialisable",
        )

    if actual == expected:
        return RunOutcome(ExecutionOutcome.PASSED, stdout=str(actual))
    return RunOutcome(
        ExecutionOutcome.FAILED,
        stdout=str(actual),
        error=f"expected {expected!r}, got {actual!r}",
    )


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " …[truncated]"
