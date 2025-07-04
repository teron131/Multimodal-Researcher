from typing import List, Optional

from pydantic import BaseModel, Field


# ===== Input and output states =====
class GraphInput(BaseModel):
    """State for the DISPLAY"""

    topic: Optional[str] = None
    video_url: Optional[str] = None


class GraphOutput(BaseModel):
    """State for the DISPLAY"""

    report: Optional[str] = None
    synthesis_text: Optional[str] = None


# ===== Plan states =====
# class Subsection(BaseModel):
#     title: str
#     description: str


class Section(BaseModel):
    title: str
    description: str
    # subsections: Optional[List[Subsection]] = None


class Plan(BaseModel):
    sections: List[Section]


# ===== Graph state =====
class GraphState(BaseModel):
    """State for the research and report generation workflow"""

    # Input fields - flatten to avoid nested state issues
    topic: Optional[str] = None
    video_url: Optional[str] = None

    # Intermediate results
    plan: Optional[Plan] = None
    search_text: Optional[str] = None
    search_sources_text: Optional[str] = None
    video_text: Optional[str] = None

    # Output fields - flatten to avoid nested state issues
    report: Optional[str] = None
    synthesis_text: Optional[str] = None
