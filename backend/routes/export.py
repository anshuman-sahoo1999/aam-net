from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
import io
import json
import csv
import zipfile
from backend.utils.session_store import InMemoryStore

router = APIRouter()

@router.get("/export")
async def export_results(
    session_id: str = Query(..., description="The session ID of the processed image"),
    format: str = Query("zip", description="Export format: 'zip', 'csv', or 'json'")
):
    session = InMemoryStore.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or has expired.")
        
    features = session.get("features", [])
    angle_maps = session.get("angle_maps", [])
    classifications = session.get("classifications", [])
    visualizations = session.get("visualizations", {})
    filename = session.get("filename", "handloom_textile.png")
    
    if not features:
        raise HTTPException(status_code=400, detail="No processed data available to export.")
        
    # Build core tabular data
    export_data = []
    for f in features:
        motif_id = f["motif_id"]
        angle_info = next((am for am in angle_maps if am["motif_id"] == motif_id), {})
        class_info = next((cl for cl in classifications if cl["motif_id"] == motif_id), {})
        
        export_data.append({
            "motif_id": motif_id,
            "area": f["area"],
            "perimeter": f["perimeter"],
            "circularity": f["circularity"],
            "texture_score": f["texture_score"],
            "orientation_angle_deg": angle_info.get("angle_degree", 0.0),
            "centroid_x": angle_info.get("x_coordinate", 0.0),
            "centroid_y": angle_info.get("y_coordinate", 0.0),
            "predicted_class": class_info.get("predicted_class", "Unclassified"),
            "confidence_score": class_info.get("confidence_score", 0.0)
        })
        
    # Return JSON format directly
    if format == "json":
        json_content = json.dumps(export_data, indent=2)
        # Clean up session to satisfy "deleted automatically after processing"
        InMemoryStore.delete_session(session_id)
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=motif_report_{session_id}.json"}
        )
        
    # Return CSV format directly
    elif format == "csv":
        csv_buffer = io.StringIO()
        if export_data:
            writer = csv.DictWriter(csv_buffer, fieldnames=export_data[0].keys())
            writer.writeheader()
            writer.writerows(export_data)
        csv_content = csv_buffer.getvalue()
        # Clean up session to satisfy "deleted automatically after processing"
        InMemoryStore.delete_session(session_id)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=motif_report_{session_id}.csv"}
        )
        
    # Return ZIP archive containing CSV, JSON, and PNG visualizations
    elif format == "zip":
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Add JSON report
            json_content = json.dumps(export_data, indent=2)
            zip_file.writestr("report.json", json_content)
            
            # 2. Add CSV report
            csv_str_buffer = io.StringIO()
            if export_data:
                writer = csv.DictWriter(csv_str_buffer, fieldnames=export_data[0].keys())
                writer.writeheader()
                writer.writerows(export_data)
            zip_file.writestr("report.csv", csv_str_buffer.getvalue())
            
            # 3. Add cached PNG Visualizations
            if "contour_overlay" in visualizations:
                zip_file.writestr("contour_overlay.png", visualizations["contour_overlay"])
            if "orientation_vectors" in visualizations:
                zip_file.writestr("orientation_vectors.png", visualizations["orientation_vectors"])
            if "angle_heatmap" in visualizations:
                zip_file.writestr("angle_heatmap.png", visualizations["angle_heatmap"])
                
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()
        
        # Clean up session to satisfy "deleted automatically after processing"
        InMemoryStore.delete_session(session_id)
        
        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=motif_analysis_{session_id}.zip"}
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Supported formats: 'zip', 'csv', 'json'.")
