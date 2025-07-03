from typing import Optional

from pydantic import BaseModel


class ResearchStateInput(BaseModel):
    """State for the research and report generation workflow input"""

    topic: Optional[str] = None
    video_url: Optional[str] = None


class ResearchStateOutput(BaseModel):
    """State for the research and report generation workflow output"""

    report: Optional[str] = None


class ResearchState(BaseModel):
    """State for the research and report generation workflow"""

    # Input fields - optional to allow partial updates
    topic: Optional[str] = None
    video_url: Optional[str] = None

    # Intermediate results
    search_text: Optional[str] = None
    search_sources_text: Optional[str] = None
    video_text: Optional[str] = None

    # Final outputs
    report: Optional[str] = None
    synthesis_text: Optional[str] = None
