"""Add image metadata columns to inspections

Revision ID: 0002_add_image_metadata
Revises: 0001_initial_schema
Create Date: 2026-11-16 18:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0002_add_image_metadata'
down_revision: Union[str, None] = '0001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('inspections', sa.Column('original_filename', sa.String(length=255), nullable=True))
    op.add_column('inspections', sa.Column('file_size_bytes', sa.Integer(), nullable=True))
    op.add_column('inspections', sa.Column('image_width', sa.Integer(), nullable=True))
    op.add_column('inspections', sa.Column('image_height', sa.Integer(), nullable=True))
    op.add_column('inspections', sa.Column('mime_type', sa.String(length=50), nullable=True))
    op.add_column('inspections', sa.Column('extra_metadata', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('inspections', 'extra_metadata')
    op.drop_column('inspections', 'mime_type')
    op.drop_column('inspections', 'image_height')
    op.drop_column('inspections', 'image_width')
    op.drop_column('inspections', 'file_size_bytes')
    op.drop_column('inspections', 'original_filename')
