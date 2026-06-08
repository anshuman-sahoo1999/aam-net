from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import cv2
import base64
from backend.utils.session_store import InMemoryStore
from backend.services.pca_mapper import PCAMapperService
from backend.utils.visualization import VisualizationUtil

router = APIRouter()

class AngleMapRequest(BaseModel):
    session_id: str

@router.post("/angle-map")
async def angle_map(request: AngleMapRequest):
    session = InMemoryStore.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or has expired.")
        
    motifs = session.get("motifs", [])
    features = session.get("features", [])
    original_img = session.get("original_img")
    
    if not motifs:
        raise HTTPException(status_code=400, detail="No motifs found in session.")
        
    try:
        # 1. Calculate PCA Orientation Mapping
        angle_maps = PCAMapperService.map_motifs_angles(motifs)
        
        # 2. Update dominant orientation inside features
        updated_features = []
        for feat in features:
            match = next((am for am in angle_maps if am["motif_id"] == feat["motif_id"]), None)
            if match:
                feat["dominant_orientation"] = match["angle_degree"]
            updated_features.append(feat)
            
        # Update session features
        InMemoryStore.update_session(request.session_id, {
            "features": updated_features,
            "angle_maps": angle_maps
        })
        
        # 3. Generate Visualizations
        overlay_img = VisualizationUtil.draw_contour_overlay(original_img, motifs)
        vectors_img = VisualizationUtil.draw_orientation_vectors(original_img, angle_maps)
        heatmap_img = VisualizationUtil.generate_angle_heatmap(original_img, motifs, angle_maps)
        dist_plot_base64 = VisualizationUtil.generate_feature_distribution(angle_maps, updated_features)
        
        # Encode cv2 images to base64
        _, enc_overlay = cv2.imencode(".png", overlay_img)
        _, enc_vectors = cv2.imencode(".png", vectors_img)
        _, enc_heatmap = cv2.imencode(".png", heatmap_img)
        
        overlay_b64 = f"data:image/png;base64,{base64.b64encode(enc_overlay).decode('utf-8')}"
        vectors_b64 = f"data:image/png;base64,{base64.b64encode(enc_vectors).decode('utf-8')}"
        heatmap_b64 = f"data:image/png;base64,{base64.b64encode(enc_heatmap).decode('utf-8')}"
        
        visualizations = {
            "contour_overlay": overlay_b64,
            "orientation_vectors": vectors_b64,
            "angle_heatmap": heatmap_b64,
            "feature_distribution": dist_plot_base64
        }
        
        # Cache visualizations in session for export
        InMemoryStore.update_session(request.session_id, {
            "visualizations": {
                "contour_overlay": enc_overlay.tobytes(),
                "orientation_vectors": enc_vectors.tobytes(),
                "angle_heatmap": enc_heatmap.tobytes()
            }
        })
        
        return {
            "session_id": request.session_id,
            "angle_maps": angle_maps,
            "visualizations": visualizations
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PCA orientation mapping failed: {str(e)}")
