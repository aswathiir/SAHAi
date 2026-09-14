"""Attach a credential to a learner id, or list the ones that have none.

    python scripts/mint_token.py --list
    python scripts/mint_token.py --learner me --name "Aswathi"
    python scripts/mint_token.py --name "Aswathi"          # fresh learner id

Two jobs, both of which the HTTP API deliberately cannot do:

* **Migration.** `x-learner-id` accepted any string, so the database already
  holds learners like `me`, `smoke-learner` and whatever the NeetCode importer
  was pointed at. None of them have a token, and with the header gone they are
  otherwise unreachable. `--learner` attaches a credential to an existing id,
  keeping its mastery, observations and session history.
* **Recovery.** A token is shown once and stored only as a SHA-256, so there is
  no "resend my token" endpoint — an endpoint that can re-read a credential is
  an endpoint that can leak one. Re-minting is an operator action, taken at the
  database, by someone who already has the database.

`--learner` is why `identity.register()` accepts an id at all, and why nothing
on the HTTP surface passes one.

Run it wherever the database is reachable:

    docker compose exec gateway python /srv/scripts/mint_token.py --list

or locally with SAHAI_GATEWAY_DSN pointed at the same Postgres.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "gateway"))

from app import identity  # noqa: E402


async def _list() -> int:
    from sqlalchemy import select

    async with identity.SessionLocal() as db:
        rows = (await db.execute(select(identity.LearnerIdentity))).scalars().all()
    if not rows:
        print("no learner identities yet")
        return 0
    print(f"{'learner_id':<24} {'display name':<24} {'last seen':<26} created")
    for r in sorted(rows, key=lambda r: r.created_at):
        seen = r.last_seen_at.isoformat() if r.last_seen_at else "never"
        print(f"{r.learner_id:<24} {r.display_name:<24} {seen:<26} {r.created_at.isoformat()}")
    return 0


async def _mint(learner: str | None, name: str) -> int:
    from sqlalchemy import select

    if learner:
        async with identity.SessionLocal() as db:
            existing = (
                await db.execute(
                    select(identity.LearnerIdentity).where(
                        identity.LearnerIdentity.learner_id == learner
                    )
                )
            ).scalar_one_or_none()
        if existing is not None:
            # Refuse rather than overwrite: two tokens for one learner is a
            # reasonable thing to want and silently invalidating the first one
            # is not. Deleting the row is an explicit act.
            print(
                f"error: {learner!r} already has a credential (created "
                f"{existing.created_at.isoformat()}). Delete that row first if "
                f"you mean to replace it.",
                file=sys.stderr,
            )
            return 1

    async with identity.SessionLocal() as db:
        learner_id, token = await identity.register(db, name, learner_id=learner)

    print(f"learner_id   {learner_id}")
    print(f"display name {name}")
    print(f"token        {token}")
    print()
    print("Shown once. Only its SHA-256 is stored — there is no way to read it back.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="show existing identities")
    parser.add_argument("--learner", help="attach to this existing learner id")
    parser.add_argument("--name", help="display name for a new credential")
    args = parser.parse_args()

    if args.list:
        return asyncio.run(_list())
    if not args.name:
        parser.error("--name is required unless --list is given")

    print(f"database: {os.getenv('SAHAI_GATEWAY_DSN', identity.DATABASE_URL)}", file=sys.stderr)
    return asyncio.run(_mint(args.learner, args.name))


if __name__ == "__main__":
    raise SystemExit(main())
