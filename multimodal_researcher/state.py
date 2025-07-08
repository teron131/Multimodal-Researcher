"""State for the research and report generation workflow"""

from typing import Optional

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
    sections: list[Section]


# ===== Search states =====
class SearchResult(BaseModel):
    section: Section
    answer: str
    sources: list[tuple[str, str]]


class SearchResults(BaseModel):
    section_results: list[SearchResult]


# ===== Video states =====
class VideoResult(BaseModel):
    video_url: str
    video_title: str
    detailed_note: str
    summary: str


class VideoResults(BaseModel):
    video_results: list[VideoResult]


# ===== Graph state =====
class GraphState(BaseModel):
    """State for the research and report generation workflow"""

    # Input fields - flatten to avoid nested state issues
    topic: Optional[str] = None
    video_urls_raw: Optional[str] = None  # Raw input
    video_urls: Optional[list[str]] = None

    # Intermediate results
    plan: Optional[Plan] = None
    search_results: Optional[SearchResults] = None
    video_results: Optional[VideoResults] = None

    # Output fields - flatten to avoid nested state issues
    report: Optional[str] = None
    synthesis_text: Optional[str] = None
