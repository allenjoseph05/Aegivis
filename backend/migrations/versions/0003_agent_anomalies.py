"""Agent anomalies table for persisting detected anomaly flags.

Revision ID: 0003
Revises: 0002
Create Date: 2026-02-22
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_anomalies (
            id              BIGSERIAL PRIMARY KEY,
            session_id      TEXT NOT NULL,
            agent_id        TEXT NOT NULL,
            org_id          TEXT NOT NULL DEFAULT '',
            rule_id         TEXT NOT NULL,
            severity        TEXT NOT NULL,
            description     TEXT NOT NULL,
            event_id        TEXT,
            sequence_number INTEGER,
            metadata        JSONB DEFAULT '{}',
            detected_at     TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_anomalies_session
            ON agent_anomalies(session_id, detected_at DESC)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_anomalies_org_time
            ON agent_anomalies(org_id, detected_at DESC)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_anomalies_severity
            ON agent_anomalies(org_id, severity, detected_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_anomalies_severity")
    op.execute("DROP INDEX IF EXISTS idx_anomalies_org_time")
    op.execute("DROP INDEX IF EXISTS idx_anomalies_session")
    op.execute("DROP TABLE IF EXISTS agent_anomalies")
