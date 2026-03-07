"""Add capability_manifests table for Phase 12 capability manifest enforcement.

Revision ID: 0013
Revises: 0012
"""
from alembic import op
import sqlalchemy as sa

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS capability_manifests (
            id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            manifest_id     TEXT        NOT NULL UNIQUE,
            org_id          TEXT        NOT NULL,
            agent_id        TEXT        NOT NULL,
            issued_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at      TIMESTAMPTZ,
            revoked_at      TIMESTAMPTZ,
            version         INTEGER     NOT NULL DEFAULT 1,
            permitted_tools JSONB       NOT NULL DEFAULT '[]'::jsonb,
            permitted_network_destinations JSONB NOT NULL DEFAULT '[]'::jsonb,
            ifc_policy      JSONB       NOT NULL DEFAULT '{}'::jsonb,
            manifest_json   TEXT        NOT NULL,
            signature       TEXT        NOT NULL,
            created_by      TEXT        NOT NULL DEFAULT 'system',
            notes           TEXT
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_capability_manifests_agent
        ON capability_manifests (org_id, agent_id)
        WHERE revoked_at IS NULL
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_capability_manifests_expires
        ON capability_manifests (expires_at)
        WHERE revoked_at IS NULL AND expires_at IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS capability_manifests")
