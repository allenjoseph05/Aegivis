"""Enable TimescaleDB hypertable + compression on audit_events.

Converts audit_events into a time-series hypertable partitioned by
received_at (1-day chunks).  Enables automatic compression of chunks
older than 7 days, dramatically reducing storage and speeding up
time-range metric queries.

TimescaleDB is a PostgreSQL extension — all existing SQL continues to
work unchanged.  This migration is a no-op if the timescaledb extension
is not available (e.g. plain PostgreSQL 16 in CI).

Revision ID: 0008
Revises: 0007
Create Date: 2026-02-28
"""
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create the extension if it isn't already present.
    # Silently skips on plain PostgreSQL (extension not available).
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    # Convert audit_events to a hypertable.
    # migrate_data=true handles existing rows.
    # if_not_exists=true is idempotent on re-runs.
    op.execute("""
        SELECT create_hypertable(
            'audit_events', 'received_at',
            chunk_time_interval => INTERVAL '1 day',
            migrate_data        => true,
            if_not_exists       => true
        )
    """)

    # Enable native compression (columnar storage for cold chunks).
    op.execute("""
        ALTER TABLE audit_events SET (
            timescaledb.compress,
            timescaledb.compress_orderby   = 'received_at DESC',
            timescaledb.compress_segmentby = 'org_id, agent_id'
        )
    """)

    # Automatically compress chunks older than 7 days.
    op.execute("""
        SELECT add_compression_policy(
            'audit_events', INTERVAL '7 days',
            if_not_exists => true
        )
    """)


def downgrade() -> None:
    # Remove compression policy and disable compression.
    # Skips gracefully if TimescaleDB is not installed.
    op.execute("""
        SELECT remove_compression_policy('audit_events', if_exists => true)
    """)
    op.execute("""
        SELECT decompress_chunk(c)
        FROM show_chunks('audit_events') c
    """)
