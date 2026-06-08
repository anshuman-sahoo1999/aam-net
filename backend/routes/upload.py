from fastapi import APIRouter, UploadFile, File, HTTPException
import uuid
import base64
import cv2
from backend.services.image_processing import ImageProcessingService
from backend.utils.session_store import InMemoryStore

router = APIRouter()

# Max file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = ["image/png", "image/jpeg", "image/jpg"]

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # 1. Type validation
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only PNG and JPEG formats are supported.")
    
    # Read file content
    contents = await file.read()
    
    # 2. Size validation
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the 10MB limit.")
        
    try:
        # Decode image to opencv
        img = ImageProcessingService.bytes_to_cv2(contents)
        
        # Run Preprocessing
        preprocessed = ImageProcessingService.preprocess_image(img)
        
        # Detect Motifs
        motifs = ImageProcessingService.detect_motifs(preprocessed["thresh"], img)
        
        if not motifs:
            raise HTTPException(status_code=422, detail="No motifs detected in the image. Please verify contrast/features.")
            
        # Generate previews
        # Original preview resized if too large for speed
        h, w = img.shape[:2]
        max_preview_dim = 800
        if max(h, w) > max_preview_dim:
            scale = max_preview_dim / max(h, w)
            preview_orig = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            preview_orig = img
            
        _, encoded_orig = cv2.imencode(".png", preview_orig)
        _, encoded_thresh = cv2.imencode(".png", preprocessed["thresh"])
        
        orig_base64 = base64.b64encode(encoded_orig).decode('utf-8')
        thresh_base64 = base64.b64encode(encoded_thresh).decode('utf-8')
        
        # Create session
        session_id = str(uuid.uuid4())
        
        # Prepare basic motif details for response
        motifs_metadata = []
        for m in motifs:
            x, y, w, h = m["bounding_box"]
            motifs_metadata.append({
                "id": m["id"],
                "centroid": m["centroid"],
                "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                "area": float(m["area"]),
                "circularity": float(m["circularity"])
            })
            
        # Cache in memory
        InMemoryStore.create_session(session_id, {
            "filename": file.filename,
            "original_img": img,
            "preprocessed": preprocessed,
            "motifs": motifs,
            "metadata": motifs_metadata
        })
        
        return {
            "session_id": session_id,
            "motifs_count": len(motifs),
            "original_preview": f"data:image/png;base64,{orig_base64}",
            "processed_preview": f"data:image/png;base64,{thresh_base64}",
            "motifs": motifs_metadata
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image upload and processing failed: {str(e)}")
