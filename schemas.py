from pydantic import BaseModel
from typing import Optional

class JAVRequest(BaseModel):
    video_path: str
    output_report: Optional[str] = "highlight_reel_report.txt"