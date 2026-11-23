import os
import sys

# Ensure backend root directory is on sys.path
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.core.security import get_password_hash
from app.db.session import SessionLocal
from app.models.defect import Defect
from app.models.inspection import Inspection, InspectionDecision, InspectionStatus
from app.models.product import Product
from app.models.report import Report
from app.models.user import User, UserRole


def seed_database():
    print("🌱 Starting VisionInspect AI Database Seeding...")
    db = SessionLocal()

    try:
        # 1. Seed Admin User
        admin = (
            db.query(User)
            .filter(User.email == settings.FIRST_SUPERUSER_EMAIL)
            .first()
        )
        if not admin:
            admin = User(
                email=settings.FIRST_SUPERUSER_EMAIL,
                hashed_password=get_password_hash(settings.FIRST_SUPERUSER_PASSWORD),
                full_name=settings.FIRST_SUPERUSER_NAME,
                role=UserRole.ADMIN.value,
                is_active=True,
            )
            db.add(admin)
            db.commit()
            db.refresh(admin)
            print(f"✅ Admin user created: {admin.email} (Password: {settings.FIRST_SUPERUSER_PASSWORD})")
        else:
            print(f"ℹ️ Admin user already exists: {admin.email}")

        # 2. Seed Supervisor User
        supervisor = (
            db.query(User)
            .filter(User.email == "supervisor@visioninspect.ai")
            .first()
        )
        if not supervisor:
            supervisor = User(
                email="supervisor@visioninspect.ai",
                hashed_password=get_password_hash("Super@123"),
                full_name="Sarah Supervisor",
                role=UserRole.SUPERVISOR.value,
                is_active=True,
            )
            db.add(supervisor)
            db.commit()
            db.refresh(supervisor)
            print(f"✅ Supervisor user created: {supervisor.email} (Password: Super@123)")
        else:
            print(f"ℹ️ Supervisor user already exists: {supervisor.email}")

        # 3. Seed Quality Engineer User
        qe = (
            db.query(User)
            .filter(User.email == "qe@visioninspect.ai")
            .first()
        )
        if not qe:
            qe = User(
                email="qe@visioninspect.ai",
                hashed_password=get_password_hash("Engineer@123"),
                full_name="Alex Engineer",
                role=UserRole.QUALITY_ENGINEER.value,
                is_active=True,
            )
            db.add(qe)
            db.commit()
            db.refresh(qe)
            print(f"✅ Quality Engineer user created: {qe.email} (Password: Engineer@123)")
        else:
            print(f"ℹ️ Quality Engineer user already exists: {qe.email}")

        # 4. Seed Products
        p1 = db.query(Product).filter(Product.sku == "MVT-METAL-01").first()
        if not p1:
            p1 = Product(
                name="Cold-Rolled Steel Sheet",
                sku="MVT-METAL-01",
                description="Cold-rolled high tensile steel sheet for automotive stamping panels.",
                is_active=True,
            )
            db.add(p1)

        p2 = db.query(Product).filter(Product.sku == "NEU-STRIP-02").first()
        if not p2:
            p2 = Product(
                name="Hot-Rolled Steel Strip",
                sku="NEU-STRIP-02",
                description="Hot-rolled surface inspection strip compliant with NEU defect dataset standards.",
                is_active=True,
            )
            db.add(p2)

        p3 = db.query(Product).filter(Product.sku == "PCB-IND-03").first()
        if not p3:
            p3 = Product(
                name="Industrial Controller PCB-v2",
                sku="PCB-IND-03",
                description="Double-sided multi-layer printed circuit board for robotic workcells.",
                is_active=True,
            )
            db.add(p3)

        db.commit()
        db.refresh(p1)
        db.refresh(p2)
        print("✅ Sample products seeded.")

        # 5. Seed Sample Inspections & Defects
        existing_inspections = db.query(Inspection).count()
        if existing_inspections == 0:
            # Inspection 1: Pass
            insp1 = Inspection(
                product_id=p1.id,
                image_path="/data/sample_images/metal_sheet_clean_001.png",
                status=InspectionStatus.COMPLETED.value,
                decision=InspectionDecision.PASS.value,
                overall_severity=12.5,
                created_by=qe.id,
            )
            db.add(insp1)

            # Inspection 2: Critical Defect (Reject)
            insp2 = Inspection(
                product_id=p2.id,
                image_path="/data/sample_images/neu_crazing_defect_042.png",
                status=InspectionStatus.COMPLETED.value,
                decision=InspectionDecision.FAIL.value,
                overall_severity=84.2,
                created_by=qe.id,
            )
            db.add(insp2)
            db.commit()
            db.refresh(insp2)

            # Defect on Inspection 2
            defect = Defect(
                inspection_id=insp2.id,
                defect_type="crazing",
                confidence=0.94,
                bounding_box={"xmin": 120, "ymin": 85, "xmax": 340, "ymax": 280},
                segmentation_mask_path="/data/masks/neu_crazing_042_mask.png",
                size_score=85.0,
                location_score=80.0,
                defect_type_score=90.0,
                confidence_score=94.0,
                severity_score=84.2,
            )
            db.add(defect)

            # Report on Inspection 2
            report = Report(
                inspection_id=insp2.id,
                summary="Critical crazing detected across central stress plane. Component rejected.",
                notes="Recommended mold temperature recalibration on Line 3.",
                generated_by=supervisor.id,
                report_data={
                    "severity": 84.2,
                    "action": "reject",
                    "formula": "Size*0.30 + Location*0.25 + DefectType*0.25 + Confidence*0.20",
                },
            )
            db.add(report)
            db.commit()
            print("✅ Sample inspection, defect, and report records seeded.")

        print("🎉 Database seeding completed successfully!")

    except Exception as e:
        db.rollback()
        print(f"❌ Error during database seeding: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
