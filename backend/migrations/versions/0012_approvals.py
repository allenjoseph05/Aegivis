"""Add approvals table for HITL (human-in-the-loop) tool call approvals.

Revision ID: 0012
Revises: 0011
"""
from alembic import op
import sqlalchemy as sa

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS approvals (
            id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id        TEXT        NOT NULL,
            session_id    TEXT        NOT NULL,
            agent_id      TEXT        NOT NULL,
            tool_name     TEXT        NOT NULL,
            tool_args     JSONB       NOT NULL DEFAULT '{}',
            trigger       TEXT        NOT NULL,
            status        TEXT        NOT NULL DEFAULT 'pending',
            decided_by    TEXT,
            decision_note TEXT,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            decided_at    TIMESTAMPTZ,
            expires_at    TIMESTAMPTZ NOT NULL
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_approvals_pending
            ON approvals (org_id, status, created_at)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_approvals_session
            ON approvals (org_id, session_id)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS approvals")
