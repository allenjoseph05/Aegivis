"""Add missing performance indexes for common query patterns.

policy_violations(org_id, timestamp_ns) — violations list ordered by time
policy_violations(session_id)           — session detail violations lookup
approvals(approval_id)                  — HITL poll queries every 2 s
audit_events(session_id, event_type)    — reasoning trace JSONB filter query

Revision ID: 0016
Revises: 0015
"""
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None

def upgrade() -> None:
    # NOTE: CONCURRENTLY cannot run inside a transaction (Alembic default).
    # These are startup migrations applied to a small table, so regular
    # CREATE INDEX (transactional, locks table briefly) is fine here.
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pv_org_time
        ON policy_violations (org_id, timestamp_ns DESC);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pv_session
        ON policy_violations (session_id);
    """)
    # approvals table uses (id uuid) PK and already has org+status+created_at
    # and org+session_id indexes from migration 0012. No additional index needed.
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_ae_session_etype
        ON audit_events (session_id, event_type);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_pv_org_time;")
    op.execute("DROP INDEX IF EXISTS idx_pv_session;")
    op.execute("DROP INDEX IF EXISTS idx_ae_session_etype;")
