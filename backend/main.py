from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import jwt

from app.auth import create_access_token, get_password_hash, verify_password
from app.config import SECRET_KEY, ALGORITHM, UPLOAD_DIR
from app.database import Base, SessionLocal, engine, get_db
from app.models import Inspection, User
from app.schemas import InspectionResult, TokenResponse, UserCreate, UserLogin
from app.services.model_service import simple_defect_predict


def seed_default_user(db: Session):
    default_email = "admin@example.com"
    default_password = "admin123"
    user = db.query(User).filter(User.email == default_email).first()
    if user is None:
        db.add(
            User(
                full_name="Admin User",
                email=default_email,
                password_hash=get_password_hash(default_password),
                role="admin",
            )
        )
        db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_default_user(db)
    yield


app = FastAPI(title="VisionInspect AI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@app.get("/")
def home():
    return {"message": "VisionInspect AI backend is running", "status": "ok"}


@app.post("/register", response_model=TokenResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    user_data = User(
        full_name=user.full_name,
        email=str(user.email),
        password_hash=get_password_hash(user.password),
        role=user.role,
    )
    db.add(user_data)
    db.commit()
    db.refresh(user_data)

    access_token = create_access_token({"sub": user_data.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_data.id,
            "full_name": user_data.full_name,
            "email": user_data.email,
            "role": user_data.role,
        },
    }


@app.post("/login", response_model=TokenResponse)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == str(user.email)).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        if str(user.email) == "admin@example.com" and user.password == "admin123":
            db_user = db.query(User).filter(User.email == "admin@example.com").first()
            if db_user is None:
                seed_default_user(db)
                db_user = db.query(User).filter(User.email == "admin@example.com").first()
        if not db_user or not verify_password(user.password, db_user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token({"sub": db_user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": db_user.id,
            "full_name": db_user.full_name,
            "email": db_user.email,
            "role": db_user.role,
        },
    }


@app.post("/upload-image")
def upload_image(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    safe_name = file.filename.replace(" ", "_")
    upload_path = UPLOAD_DIR / safe_name
    with upload_path.open("wb") as buffer:
        buffer.write(file.file.read())

    inspection = Inspection(
        user_id=current_user.id,
        file_name=safe_name,
        file_path=str(upload_path),
        status="uploaded",
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    return {
        "message": "Image uploaded successfully",
        "inspection_id": inspection.id,
        "file_name": inspection.file_name,
        "file_path": inspection.file_path,
    }


@app.post("/predict/{inspection_id}")
def predict_inspection(inspection_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id, Inspection.user_id == current_user.id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")

    result = simple_defect_predict(inspection.file_path)

    inspection.status = result["status"]
    inspection.defect_type = result["defect_type"]
    inspection.confidence = result["confidence"]
    inspection.severity = result["severity"]
    inspection.notes = result["notes"]
    db.commit()

    return {
        "inspection_id": inspection.id,
        "status": inspection.status,
        "defect_type": inspection.defect_type,
        "confidence": inspection.confidence,
        "severity": inspection.severity,
        "notes": inspection.notes,
    }


@app.get("/inspections", response_model=List[InspectionResult])
def get_inspections(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    inspections = db.query(Inspection).filter(Inspection.user_id == current_user.id).order_by(Inspection.id.desc()).all()
    return inspections


@app.get("/inspections/{inspection_id}", response_model=InspectionResult)
def get_inspection(inspection_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id, Inspection.user_id == current_user.id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")
    return inspection


@app.get("/dashboard")
def dashboard(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    total = db.query(Inspection).filter(Inspection.user_id == current_user.id).count()
    defective = db.query(Inspection).filter(Inspection.user_id == current_user.id, Inspection.status == "defective").count()
    normal = db.query(Inspection).filter(Inspection.user_id == current_user.id, Inspection.status == "normal").count()

    return {
        "total_inspections": total,
        "defective_count": defective,
        "normal_count": normal,
        "user": current_user.email,
    }
