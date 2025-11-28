import os
from fastapi import APIRouter, HTTPException
from app.schemas import JAVRequest
# 引入业务逻辑
from app.services.stocking_logic import execute_stocking_scan

router = APIRouter()


@router.post("/jav-stocking")
def scan_stocking(req: JAVRequest):
    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    try:
        # execute_stocking_scan 现在返回的是 merge 好的详细字典列表
        final_segments = execute_stocking_scan(req.video_path)

        return {
            "status": "success",
            "logic_version": "v1_stocking_weighted",
            "video": req.video_path,
            "found_segments": len(final_segments),
            # 🔥 修改点：直接返回 final_segments，不要用 for 循环去解包
            "timeline": final_segments
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))