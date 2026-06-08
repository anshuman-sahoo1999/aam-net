from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.utils.session_store import InMemoryStore
from backend.services.classifier import ClassifierService

router = APIRouter()

class ClassifyRequest(BaseModel):
    session_id: str
    model_type: str = "Random Forest"  # Options: "Random Forest", "SVM", "XGBoost"

@router.post("/classify")
async def classify_motifs(request: ClassifyRequest):
    session = InMemoryStore.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or has expired.")
        
    features = session.get("features", [])
    if not features:
        raise HTTPException(status_code=400, detail="Please extract features before running classification.")
        
    if request.model_type not in ["Random Forest", "SVM", "XGBoost"]:
        raise HTTPException(status_code=400, detail="Invalid model_type. Must be 'Random Forest', 'SVM', or 'XGBoost'.")
        
    try:
        classifications = []
        for feat in features:
            cls_result = ClassifierService.classify_motif(feat, request.model_type)
            classifications.append({
                "motif_id": feat["motif_id"],
                "predicted_class": cls_result["predicted_class"],
                "confidence_score": cls_result["confidence_score"],
                "probabilities": cls_result["probabilities"]
            })
            
        # Update session store
        InMemoryStore.update_session(request.session_id, {"classifications": classifications})
        
        # Calculate dashboard metrics (analytics)
        angles = [f.get("dominant_orientation", 0.0) for f in features]
        avg_angle = float(sum(angles) / len(angles)) if angles else 0.0
        
        # Most frequent motif type
        counts = {}
        for c in classifications:
            counts[c["predicted_class"]] = counts.get(c["predicted_class"], 0) + 1
        most_frequent = max(counts, key=counts.get) if counts else "N/A"
        
        analytics = {
            "total_motifs": len(features),
            "average_angle": avg_angle,
            "most_frequent_class": most_frequent,
            "class_distribution": counts
        }
        
        InMemoryStore.update_session(request.session_id, {"analytics": analytics})
        
        return {
            "session_id": request.session_id,
            "classifications": classifications,
            "analytics": analytics
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
