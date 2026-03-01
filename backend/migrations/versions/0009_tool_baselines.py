"""Add agent_tool_baselines table for per-agent tool approval workflow.

The proxy auto-discovers every tool in the tools[] array and inserts a row
here on first sight.  Operators review the table in the dashboard and set
status = 'approved' or 'denied'.  The proxy then enforces the approved set
at TOOL_CALL_START time (blocking unapproved tools).

Status lifecycle:
  pending  → (operator review) → approved | denied

Revision ID: 0009
Revises: 0008
Create Date: 2026-03-01
"""
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_tool_baselines (
            id            SERIAL PRIMARY KEY,
            org_id        TEXT NOT NULL,
            agent_id      TEXT NOT NULL,
            tool_name     TEXT NOT NULL,
            -- Full tool definition stored as JSON so operators can read the
            -- description and parameter schema in the dashboard.
            tool_schema   JSONB NOT NULL DEFAULT '{}',
            first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            -- Cumulative call count across all sessions for this tool.
            call_count    INTEGER NOT NULL DEFAULT 1,
            -- 'pending' = discovered but not yet reviewed by an operator.
            -- 'approved' = operator has explicitly allowed this tool.
            -- 'denied'   = operator has explicitly blocked this tool.
            status        TEXT NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending', 'approved', 'denied')),
            approved_at   TIMESTAMPTZ,
            UNIQUE (org_id, agent_id, tool_name)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_atb_org_agent
            ON agent_tool_baselines (org_id, agent_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_atb_status
            ON agent_tool_baselines (org_id, agent_id, status)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_tool_baselines")
