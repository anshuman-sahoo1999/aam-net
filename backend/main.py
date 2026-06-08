from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import os

from backend.routes import upload, features, angle_map, classify, export
from backend.services.classifier import ClassifierService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize/train the ML classification models on startup
    ClassifierService.initialize_models()
    yield
    # Cleanup logic (if any) goes here

app = FastAPI(
    title="Motif Feature Extraction and Angle Mapping System for Handloom Textiles",
    description="Automated computer vision system to segment, extract, map orientation angles, and classify motifs in handloom fabric images.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers directly at the root as specified in the API design
app.include_router(upload.router, tags=["Upload"])
app.include_router(features.router, tags=["Features"])
app.include_router(angle_map.router, tags=["Angle Mapping"])
app.include_router(classify.router, tags=["Classification"])
app.include_router(export.router, tags=["Export"])

# Mount static frontend files if directory exists
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
