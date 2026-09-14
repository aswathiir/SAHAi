"""Who the caller is — verified, not claimed.

Until now `x-learner-id` was a header the browser set to whatever it liked, and
`_learner()` returned it unchanged. Every ownership check downstream compares
against that value, so all of them rested on the client's honesty: sending
`x-learner-id: someone-else` was enough to read another learner's sessions,
mastery and submission history.

This terminates identity at the gateway, which is the only service the host can
reach, and is what the gateway's own docstring always said should happen.

**Bearer tokens, not passwords.** Registration mints a 256-bit random token and
returns it once; the row keeps only its SHA-256. The project stores no
passwords and never sees one, which removes the entire category of problem that
comes with storing them. A fast hash is the right choice *here* specifically:
bcrypt and argon2 exist to make offline brute force expensive against
low-entropy human secrets, and a 32-byte `secrets.token_urlsafe` has no
brute-force surface to protect. Lookup is by exact hash, so there is no
comparison to time-attack either.

**What this is not.** It authenticates a *bearer*, not a person: anyone holding
the token is the learner, there is no second factor, no recovery, and no
revocation beyond deleting the row. For a system with no email, no password
reset and no real user base that is the honest shape. Losing the token means
minting a new one with `scripts/mint_token.py`.

**Where trust stops.** The gateway forwards the verified id to session and
tracer as `x-learner-id`, and those services still trust that header — but they
are on an internal Docker network with no published ports, so the header is
only settable by something that already has network access to them. That is a
topology assumption, not a cryptographic one, and it is why the compose file's
two-network split matters. Signing the internal hop would make it cryptographic;
it is not done here, and the assumption is written down instead of implied.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = os.getenv(
    "SAHAI_GATEWAY_DSN", "postgresql+asyncpg://sahai:sahai@postgres:5432/sahai"
)

TOKEN_BYTES = 32
LEARNER_ID_PREFIX = "lnr_"
MAX_DISPLAY_NAME = 80


class Base(DeclarativeBase):
    pass


class LearnerIdentity(Base):
    __tablename__ = "learner_identity"

    # Server-generated, never taken from the request. Letting a caller choose
    # its own id would let it claim an id that already holds someone's mastery
    # history, which is the same hole in a different place.
    learner_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(MAX_DISPLAY_NAME))
    # Unique so a (vanishingly unlikely) token collision fails loudly at insert
    # rather than silently giving two learners the same credential.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def new_learner_id() -> str:
    return f"{LEARNER_ID_PREFIX}{secrets.token_hex(8)}"


async def register(db: AsyncSession, display_name: str, learner_id: str | None = None) -> tuple[str, str]:
    """Create an identity and return `(learner_id, token)`.

    The token is returned exactly once and is not recoverable afterwards —
    only its hash is stored.

    `learner_id` is for `scripts/mint_token.py` only, so an operator can attach
    a credential to a learner that already has history (an imported NeetCode
    profile, a seeded demo account). It is never taken from an HTTP request.
    """
    token = new_token()
    row = LearnerIdentity(
        learner_id=learner_id or new_learner_id(),
        display_name=display_name.strip()[:MAX_DISPLAY_NAME],
        token_hash=hash_token(token),
        created_at=datetime.now(timezone.utc),
    )
    db.add(row)
    await db.commit()
    return row.learner_id, token


async def resolve(db: AsyncSession, token: str) -> LearnerIdentity | None:
    """The identity this token belongs to, or None.

    `last_seen_at` is written on every successful call. It is the only way to
    tell an abandoned credential from a live one, and it costs one UPDATE on a
    primary-key row.
    """
    row = (
        await db.execute(
            select(LearnerIdentity).where(LearnerIdentity.token_hash == hash_token(token))
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    row.last_seen_at = datetime.now(timezone.utc)
    await db.commit()
    return row
