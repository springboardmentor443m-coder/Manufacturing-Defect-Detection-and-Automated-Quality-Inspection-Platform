"""Add image processing and quality columns

Revision ID: 0003_image_proc_quality
Revises: 0002_add_image_metadata
Create Date: 2026-11-18 01:55:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "0003_image_proc_quality"
down_revision = "0002_add_image_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("inspections", sa.Column("processed_image_path", sa.String(length=1024), nullable=True))
    op.add_column("inspections", sa.Column("quality_score", sa.Float(), nullable=True))
    op.add_column("inspections", sa.Column("quality_verdict", sa.String(length=50), nullable=True))
    op.add_column("inspections", sa.Column("quality_report", sa.JSON(), nullable=True))
    op.add_column("inspections", sa.Column("preprocessing_config", sa.JSON(), nullable=True))
    op.create_index(op.f("ix_inspections_quality_verdict"), "inspections", ["quality_verdict"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_inspections_quality_verdict"), table_name="inspections")
    op.drop_column("inspections", "preprocessing_config")
    op.drop_column("inspections", "quality_report")
    op.drop_column("inspections", "quality_verdict")
    op.drop_column("inspections", "quality_score")
    op.drop_column("inspections", "processed_image_path")
