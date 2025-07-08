"""Utility functions for the research and report generation workflow"""

import re

from google.genai import types
from rich.console import Console
from rich.markdown import Markdown

from multimodal_researcher.state import GraphState


def extract_youtube_video_urls(text: str) -> list[str]:
    """Extract all YouTube video IDs from a string containing multiple URLs."""
    text = text.strip()

    video_ids = []

    # Pattern for youtu.be/VIDEO_ID
    youtu_be_pattern = r"youtu\.be/([a-zA-Z0-9_-]+)"

    # Pattern for youtube.com/watch?v=VIDEO_ID
    youtube_com_pattern = r"youtube\.com/watch\?v=([a-zA-Z0-9_-]+)(?:&|$)"

    # Find all youtu.be matches
    youtu_be_matches = re.findall(youtu_be_pattern, text)
    video_ids.extend(youtu_be_matches)

    # Find all youtube.com matches
    youtube_com_matches = re.findall(youtube_com_pattern, text)
    video_ids.extend(youtube_com_matches)

    video_urls = [f"https://youtu.be/{video_id}" for video_id in video_ids]

    return video_urls


def create_report_prompt(state: GraphState) -> str:
    report_prompt = f"""Write a report about the topic {state.topic}.

It consists of the following sections:

"""
    for section in state.section_results.section_results:
        report_prompt += f"""SECTION {section.section.index}: {section.section.title}

DESCRIPTION:
{section.section.description}

SUGGESTED ANSWER:
{section.answer}

"""

    report_prompt += f"""{'-'*50}
There are additional resources from videos (not necessarily relevant):

"""

    for video in state.video_results.video_results:
        report_prompt += f"""VIDEO: {video.video_title}

DETAILED NOTE:
{video.detailed_note}

SUMMARY:
{video.summary}

"""

    report_prompt += f"""{'-'*50}
Please create a comprehensive report that:
1. Identifies key themes and insights from both sources
2. Highlights any complementary or contrasting perspectives
3. Provides an overall analysis of the topic based on this multi-modal research
4. Keep it concise but thorough (3-4 paragraphs)
Focus on creating a coherent narrative that brings together the best insights from both sources."""

    return report_prompt


def extract_search_response(response: types.GenerateContentResponse):
    """Extract text from Gemini search response and return text with sources"""
    # Extract main content
    text = response.text

    # Get candidate for grounding metadata
    candidate = response.candidates[0]

    # Build sources text block
    sources_text = ""

    # Extract grounding metadata if available
    if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
        # Extract and collect source URLs
        if candidate.grounding_metadata.grounding_chunks:
            sources_list = []
            for i, chunk in enumerate(candidate.grounding_metadata.grounding_chunks, 1):
                if hasattr(chunk, "web") and chunk.web:
                    title = getattr(chunk.web, "title", "No title") or "No title"
                    uri = getattr(chunk.web, "uri", "No URI") or "No URI"
                    sources_list.append(f"{i}. {title}\n   {uri}")

            sources_text = "\n".join(sources_list)

    return text, sources_text


def display_gemini_search_response(response: types.GenerateContentResponse):
    """Extract text from Gemini search response and display as markdown with references"""
    console = Console()

    # Extract main content
    text = response.text
    md = Markdown(text)
    console.print(md)

    # Get candidate for grounding metadata
    candidate = response.candidates[0]

    # Build sources text block
    sources_text = ""

    # Display grounding metadata if available
    if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
        console.print("\n" + "=" * 50)
        console.print("[bold blue]References & Sources[/bold blue]")
        console.print("=" * 50)

        # Display and collect source URLs
        if candidate.grounding_metadata.grounding_chunks:
            console.print(f"\n[bold]Sources ({len(candidate.grounding_metadata.grounding_chunks)}):[/bold]")
            sources_list = []
            for i, chunk in enumerate(candidate.grounding_metadata.grounding_chunks, 1):
                if hasattr(chunk, "web") and chunk.web:
                    title = getattr(chunk.web, "title", "No title") or "No title"
                    uri = getattr(chunk.web, "uri", "No URI") or "No URI"
                    console.print(f"{i}. {title}")
                    console.print(f"   [dim]{uri}[/dim]")
                    sources_list.append(f"{i}. {title}\n   {uri}")

            sources_text = "\n".join(sources_list)

        # Display grounding supports (which text is backed by which sources)
        if candidate.grounding_metadata.grounding_supports:
            console.print(f"\n[bold]Text segments with source backing:[/bold]")
            for support in candidate.grounding_metadata.grounding_supports[:5]:  # Show first 5
                if hasattr(support, "segment") and support.segment:
                    snippet = support.segment.text[:100] + "..." if len(support.segment.text) > 100 else support.segment.text
                    source_nums = [str(i + 1) for i in support.grounding_chunk_indices]
                    console.print(f"• \"{snippet}\" [dim](sources: {', '.join(source_nums)})[/dim]")

    return text, sources_text
