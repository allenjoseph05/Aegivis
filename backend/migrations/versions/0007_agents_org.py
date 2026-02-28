"""Add org_id column to agents table.

Allows per-org agent registries: an agent_id is now unique within
an org rather than globally.  Composite primary key (org_id, agent_id).

Revision ID: 0007
Revises: 0006
Create Date: 2026-02-28
"""
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add org_id column (existing rows default to 'default-org')
    op.execute("""
        ALTER TABLE agents
        ADD COLUMN IF NOT EXISTS org_id TEXT NOT NULL DEFAULT 'default-org'
    """)

    # Re-key as composite PK so the same agent_id can exist in multiple orgs
    op.execute("ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_pkey")
    op.execute("ALTER TABLE agents ADD PRIMARY KEY (org_id, agent_id)")

    op.execute("CREATE INDEX IF NOT EXISTS idx_agents_org ON agents(org_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_agents_org")
    op.execute("ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_pkey")
    op.execute("ALTER TABLE agents ADD PRIMARY KEY (agent_id)")
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS org_id")
