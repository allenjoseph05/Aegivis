"""Initial schema — audit_events table with append-only enforcement.

Revision ID: 0001
Revises:
Create Date: 2026-02-21
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create the partitioned audit_events table
    op.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id          TEXT PRIMARY KEY,
            schema_version    TEXT NOT NULL DEFAULT '1.0',
            org_id            TEXT NOT NULL,
            session_id        TEXT NOT NULL,
            agent_id          TEXT NOT NULL,
            provider          TEXT NOT NULL,
            model             TEXT NOT NULL,
            interception_layer TEXT NOT NULL DEFAULT 'proxy',
            run_id            TEXT NOT NULL,
            parent_run_id     TEXT,
            event_type        TEXT NOT NULL,
            payload           JSONB NOT NULL,
            payload_hash      TEXT,
            pii_detected      TEXT[] DEFAULT '{}',
            timestamp_ns      BIGINT NOT NULL,
            sequence_number   INTEGER NOT NULL,
            previous_hash     TEXT NOT NULL,
            current_hash      TEXT NOT NULL,
            received_at       TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Unique constraint on current_hash (tamper detection: two events cannot share a hash)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_current_hash ON audit_events(current_hash)
    """)

    # Performance indexes
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_session ON audit_events(session_id, sequence_number)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_org_time ON audit_events(org_id, timestamp_ns DESC)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(org_id, event_type, timestamp_ns)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_payload_gin ON audit_events USING GIN (payload)
    """)

    # Append-only enforcement: prevent UPDATE and DELETE at DB level
    op.execute("""
        CREATE OR REPLACE FUNCTION raise_immutability_error()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION 'audit_events is append-only: UPDATE and DELETE are not permitted';
        END;
        $$ LANGUAGE plpgsql
    """)

    op.execute("""
        CREATE TRIGGER no_modifications
            BEFORE UPDATE OR DELETE ON audit_events
            FOR EACH ROW EXECUTE FUNCTION raise_immutability_error()
    """)

    # Revoke UPDATE/DELETE from application user (belt + suspenders)
    # Note: this creates the user if it doesn't exist
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'abb_app') THEN
                CREATE ROLE abb_app;
            END IF;
        END $$
    """)
    op.execute("REVOKE UPDATE, DELETE ON audit_events FROM abb_app")


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS no_modifications ON audit_events")
    op.execute("DROP FUNCTION IF EXISTS raise_immutability_error()")
    op.execute("DROP TABLE IF EXISTS audit_events")
