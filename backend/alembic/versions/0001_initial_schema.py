"""Initial database schema with users, products, inspections, defects, reports

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-11-14 18:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('role', sa.String(length=50), server_default='quality_engineer', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_role'), 'users', ['role'], unique=False)

    # 2. products table
    op.create_table(
        'products',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('sku', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_products_id'), 'products', ['id'], unique=False)
    op.create_index(op.f('ix_products_name'), 'products', ['name'], unique=False)
    op.create_index(op.f('ix_products_sku'), 'products', ['sku'], unique=True)

    # 3. inspections table
    op.create_table(
        'inspections',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('product_id', sa.Integer(), nullable=False),
        sa.Column('image_path', sa.String(length=1024), nullable=False),
        sa.Column('status', sa.String(length=50), server_default='pending', nullable=False),
        sa.Column('decision', sa.String(length=50), nullable=True),
        sa.Column('overall_severity', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_inspections_id'), 'inspections', ['id'], unique=False)
    op.create_index(op.f('ix_inspections_product_id'), 'inspections', ['product_id'], unique=False)
    op.create_index(op.f('ix_inspections_created_by'), 'inspections', ['created_by'], unique=False)
    op.create_index(op.f('ix_inspections_status'), 'inspections', ['status'], unique=False)
    op.create_index(op.f('ix_inspections_decision'), 'inspections', ['decision'], unique=False)

    # 4. defects table
    op.create_table(
        'defects',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('inspection_id', sa.Integer(), nullable=False),
        sa.Column('defect_type', sa.String(length=100), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('bounding_box', sa.JSON(), nullable=True),
        sa.Column('segmentation_mask_path', sa.String(length=1024), nullable=True),
        sa.Column('size_score', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('location_score', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('defect_type_score', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('confidence_score', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('severity_score', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['inspection_id'], ['inspections.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_defects_id'), 'defects', ['id'], unique=False)
    op.create_index(op.f('ix_defects_inspection_id'), 'defects', ['inspection_id'], unique=False)
    op.create_index(op.f('ix_defects_defect_type'), 'defects', ['defect_type'], unique=False)
    op.create_index(op.f('ix_defects_severity_score'), 'defects', ['severity_score'], unique=False)

    # 5. reports table
    op.create_table(
        'reports',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('inspection_id', sa.Integer(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('generated_by', sa.Integer(), nullable=True),
        sa.Column('report_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['inspection_id'], ['inspections.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['generated_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_reports_id'), 'reports', ['id'], unique=False)
    op.create_index(op.f('ix_reports_inspection_id'), 'reports', ['inspection_id'], unique=False)
    op.create_index(op.f('ix_reports_generated_by'), 'reports', ['generated_by'], unique=False)


def downgrade() -> None:
    op.drop_table('reports')
    op.drop_table('defects')
    op.drop_table('inspections')
    op.drop_table('products')
    op.drop_table('users')
