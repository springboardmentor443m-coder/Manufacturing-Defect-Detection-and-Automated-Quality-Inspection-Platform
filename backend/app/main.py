import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.v1.api import api_router
from app.core.config import settings


def create_app() -> FastAPI:
    """Application factory for VisionInspect AI FastAPI backend."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
        description="VisionInspect AI - Manufacturing Defect Detection & Quality Inspection Platform API",
        version="1.0.0",
    )

    # Set up CORS middleware
    if settings.CORS_ORIGINS:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Mount API v1 router
    application.include_router(api_router, prefix=settings.API_V1_STR)

    # Ensure uploads directory exists and mount static files
    upload_path = Path(settings.UPLOAD_DIR).resolve()
    upload_path.mkdir(parents=True, exist_ok=True)
    application.mount("/uploads", StaticFiles(directory=str(upload_path)), name="uploads")

    @application.get("/docs", include_in_schema=False)
    def redirect_docs():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"{settings.API_V1_STR}/docs")

    @application.get("/redoc", include_in_schema=False)
    def redirect_redoc():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"{settings.API_V1_STR}/redoc")

    @application.get("/", tags=["Root"])
    def root():
        return {
            "name": settings.PROJECT_NAME,
            "version": "1.0.0",
            "docs": f"{settings.API_V1_STR}/docs",
            "health": f"{settings.API_V1_STR}/health",
        }

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
