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
    import logging
    logger = logging.getLogger(__name__)

    # Check if the timescaledb extension is available on this PostgreSQL install.
    # On plain postgres:16-alpine it is not; on timescale/timescaledb:latest-pg16 it is.
    conn = op.get_bind()
    row = conn.execute(
        __import__("sqlalchemy").text(
            "SELECT COUNT(*) FROM pg_available_extensions WHERE name = 'timescaledb'"
        )
    ).scalar()

    if not row:
        logger.warning(
            "TimescaleDB extension not available on this PostgreSQL install — "
            "skipping hypertable conversion. Switch to timescale/timescaledb:latest-pg16 "
            "in docker-compose.yml to enable time-series optimisations."
        )
        return

    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    op.execute("""
        SELECT create_hypertable(
            'audit_events', 'received_at',
            chunk_time_interval => INTERVAL '1 day',
            migrate_data        => true,
            if_not_exists       => true
        )
    """)

    op.execute("""
        ALTER TABLE audit_events SET (
            timescaledb.compress,
            timescaledb.compress_orderby   = 'received_at DESC',
            timescaledb.compress_segmentby = 'org_id, agent_id'
        )
    """)

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
