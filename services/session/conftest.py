"""Make sibling services importable for the end-to-end test.

Each service owns a top-level `app` package, so they collide on sys.path. Alias
them under distinct names rather than renaming the packages, which would make
every service inconsistent with the others.

Environment is set here, not in the test module: conftest is imported first, and
every service reads its DSN at module scope. Setting it later leaves the tracer
pointed at Postgres and the fixture fails on a DNS lookup.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="sahai-e2e-")
os.environ.setdefault("SAHAI_SESSION_DSN", f"sqlite+aiosqlite:///{_TMP}/session.db")
os.environ.setdefault("SAHAI_TRACER_DSN", f"sqlite+aiosqlite:///{_TMP}/tracer.db")
os.environ.setdefault("SAHAI_TUTOR_BACKEND", "stub")
os.environ.setdefault("SAHAI_TUTOR_URL", "http://tutor")
os.environ.setdefault("SAHAI_TRACER_URL", "http://tracer")
os.environ.setdefault("SAHAI_EXECUTOR_URL", "http://executor")

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _alias(service: str, alias: str) -> None:
    base = ROOT / "services" / service
    pkg_spec = importlib.util.spec_from_file_location(
        alias, base / "app" / "__init__.py", submodule_search_locations=[str(base / "app")]
    )
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules[alias] = pkg
    pkg_spec.loader.exec_module(pkg)

    for mod in ("backends", "sandbox", "main"):
        path = base / "app" / f"{mod}.py"
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(f"{alias}.{mod}", path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"{alias}.{mod}"] = m
        setattr(pkg, mod, m)
        spec.loader.exec_module(m)


sys.path.insert(0, str(ROOT / "libs" / "sahai-core"))
for svc in ("executor", "tracer", "tutor"):
    (ROOT / "services" / svc / "app" / "__init__.py").touch()
    _alias(svc, f"{svc}_app")
