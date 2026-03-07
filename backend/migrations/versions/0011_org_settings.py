"""Add org_settings table for configurable notification/integration settings.

Revision ID: 0011
Revises: 0010
"""
from alembic import op
import sqlalchemy as sa

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS org_settings (
            org_id     TEXT        NOT NULL,
            key        TEXT        NOT NULL,
            value      TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (org_id, key)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS org_settings")
