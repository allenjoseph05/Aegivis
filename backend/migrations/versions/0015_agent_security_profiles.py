"""Add agent_security_profiles table for per-agent security feature overrides.

Revision ID: 0015
Revises: 0014
"""
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_security_profiles (
            org_id      TEXT        NOT NULL,
            agent_id    TEXT        NOT NULL,
            key         TEXT        NOT NULL,
            value       TEXT,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (org_id, agent_id, key)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_asp_agent
        ON agent_security_profiles (org_id, agent_id)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_security_profiles")
