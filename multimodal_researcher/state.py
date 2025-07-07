"""State for the research and report generation workflow"""

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
    index: int = Field(..., description="The index of the section", ge=1, le=5)
    title: str
    description: str
    # subsections: Optional[List[Subsection]] = None


class Plan(BaseModel):
    sections: List[Section]


# ===== Search states =====
class SectionResult(BaseModel):
    section: Section
    result: str
    sources: str


class SearchResults(BaseModel):
    section_results: List[SectionResult]


# ===== Graph state =====
class GraphState(BaseModel):
    """State for the research and report generation workflow"""

    # Input fields - flatten to avoid nested state issues
    topic: Optional[str] = None
    video_url: Optional[str] = None

    # Intermediate results
    plan: Optional[Plan] = None
    search_results: Optional[SearchResults] = None
    video_text: Optional[str] = None

    # Output fields - flatten to avoid nested state issues
    report: Optional[str] = None
    synthesis_text: Optional[str] = None
