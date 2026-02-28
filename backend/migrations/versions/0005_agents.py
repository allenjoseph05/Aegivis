"""Add agents table for Agent Registry (Phase 3).

Stores registered agent identities: declared purpose, allowed tools, owner.
The proxy cross-references this table to detect unregistered agents and
tool-scope violations.

Revision ID: 0005
Revises: 0004
Create Date: 2026-02-27
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            agent_id          TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            declared_purpose  TEXT NOT NULL DEFAULT '',
            allowed_tools     JSONB NOT NULL DEFAULT '[]',
            owner             TEXT,
            registered_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agents")
