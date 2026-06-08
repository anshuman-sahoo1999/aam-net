from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.utils.session_store import InMemoryStore
from backend.services.feature_extractor import FeatureExtractorService

router = APIRouter()

class FeatureExtractionRequest(BaseModel):
    session_id: str

@router.post("/extract-features")
async def extract_features(request: FeatureExtractionRequest):
    session = InMemoryStore.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or has expired.")
        
    motifs = session.get("motifs", [])
    if not motifs:
        raise HTTPException(status_code=400, detail="No motifs found in the specified session.")
        
    try:
        features_list = []
        for motif in motifs:
            feats = FeatureExtractorService.extract_all_features(motif)
            features_list.append(feats)
            
        # Update session store
        InMemoryStore.update_session(request.session_id, {"features": features_list})
        
        # Format a clean summary to return to the user
        response_data = []
        for f in features_list:
            response_data.append({
                "motif_id": f["motif_id"],
                "area": f["area"],
                "perimeter": f["perimeter"],
                "circularity": f["circularity"],
                "texture_score": f["texture_score"],
                "dominant_orientation": f["dominant_orientation"]  # Calculated in next step
            })
            
        return {
            "session_id": request.session_id,
            "features": response_data
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
