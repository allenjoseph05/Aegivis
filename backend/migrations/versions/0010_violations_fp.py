"""Add is_false_positive column to policy_violations.

Revision ID: 0010
Revises: 0009
"""
from alembic import op
import sqlalchemy as sa

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE policy_violations
        ADD COLUMN IF NOT EXISTS is_false_positive BOOLEAN NOT NULL DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS marked_fp_at TIMESTAMP WITH TIME ZONE
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE policy_violations
        DROP COLUMN IF EXISTS is_false_positive,
        DROP COLUMN IF EXISTS marked_fp_at
    """)
