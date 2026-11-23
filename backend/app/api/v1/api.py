from fastapi import APIRouter
from app.api.v1.endpoints import auth, health, inspections, products, users

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["User Management"])
api_router.include_router(products.router, prefix="/products", tags=["Products"])
api_router.include_router(inspections.router, prefix="/inspections", tags=["Inspections"])
