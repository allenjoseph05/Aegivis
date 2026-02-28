"""Add api_keys table for multi-tenancy.

Each row maps a hashed API key to an org_id. The auth middleware
performs a DB lookup + 60-second process-local cache to resolve
the org_id for every incoming request.

Revision ID: 0006
Revises: 0005
Create Date: 2026-02-28
"""
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id       TEXT NOT NULL,
            key_hash     TEXT NOT NULL UNIQUE,
            name         TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            revoked_at   TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_org  ON api_keys(org_id)")

    # Seed default-org keys from the existing hardcoded values so existing
    # deployments continue to work without any manual provisioning.
    op.execute("""
        INSERT INTO api_keys (org_id, key_hash, name)
        VALUES
          ('default-org', encode(sha256('dev-proxy-key'::bytea), 'hex'),     'Default proxy key'),
          ('default-org', encode(sha256('dev-dashboard-key'::bytea), 'hex'), 'Default dashboard key')
        ON CONFLICT (key_hash) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_api_keys_org")
    op.execute("DROP INDEX IF EXISTS idx_api_keys_hash")
    op.execute("DROP TABLE IF EXISTS api_keys")
