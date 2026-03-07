"""Add egress_rules table for dashboard-managed network destination allowlist.

Revision ID: 0014
Revises: 0013
"""
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS egress_rules (
            id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id      TEXT        NOT NULL,
            destination TEXT        NOT NULL,
            notes       TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_by  TEXT        NOT NULL DEFAULT 'dashboard',
            UNIQUE (org_id, destination)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_egress_rules_org
        ON egress_rules (org_id)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS egress_rules")
