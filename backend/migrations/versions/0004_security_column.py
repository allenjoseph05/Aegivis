"""Add security JSONB column to audit_events.

Stores the inline enforcement scan result (injection_score, credential_detected,
encoding_score, structural_score, rce_detected, ssrf_detected, schema_violations, etc.)
alongside each event for dashboard queries and SIEM export.

Revision ID: 0004
Revises: 0003
Create Date: 2026-02-26
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add nullable security column — NULL for events that pre-date this migration
    # or for events where enforcement scanning was not run (e.g. AGENT_FINISH).
    op.execute("""
        ALTER TABLE audit_events
        ADD COLUMN IF NOT EXISTS security JSONB
    """)
    # GIN index for fast queries like:
    #   WHERE security->>'injection_label' = 'malicious'
    #   WHERE (security->>'injection_score')::float > 0.8
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_security_gin
        ON audit_events USING GIN (security)
        WHERE security IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_security_gin")
    op.execute("ALTER TABLE audit_events DROP COLUMN IF EXISTS security")
