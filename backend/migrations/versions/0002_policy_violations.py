"""Policy violations table and agent baselines table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-22
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Policy violations table — records every BLOCK/ALERT action fired
    op.execute("""
        CREATE TABLE IF NOT EXISTS policy_violations (
            id               BIGSERIAL PRIMARY KEY,
            rule_name        TEXT NOT NULL,
            action           TEXT NOT NULL,
            reason           TEXT NOT NULL,
            event_type       TEXT NOT NULL,
            session_id       TEXT NOT NULL,
            agent_id         TEXT NOT NULL,
            org_id           TEXT NOT NULL DEFAULT '',
            timestamp_ns     BIGINT NOT NULL,
            received_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_violations_session
            ON policy_violations(session_id, timestamp_ns DESC)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_violations_org_time
            ON policy_violations(org_id, timestamp_ns DESC)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_violations_rule
            ON policy_violations(rule_name, timestamp_ns DESC)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_violations_agent
            ON policy_violations(agent_id, timestamp_ns DESC)
    """)

    # Agent baselines table — running stats per (org_id, agent_id)
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_baselines (
            id                    BIGSERIAL PRIMARY KEY,
            org_id                TEXT NOT NULL,
            agent_id              TEXT NOT NULL,
            session_count         INTEGER NOT NULL DEFAULT 0,
            avg_llm_calls         FLOAT NOT NULL DEFAULT 0,
            avg_tool_calls        FLOAT NOT NULL DEFAULT 0,
            avg_session_duration_ms FLOAT NOT NULL DEFAULT 0,
            avg_latency_ms        FLOAT NOT NULL DEFAULT 0,
            avg_total_tokens      FLOAT NOT NULL DEFAULT 0,
            last_updated          TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(org_id, agent_id)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_baselines")
    op.execute("DROP TABLE IF EXISTS policy_violations")
