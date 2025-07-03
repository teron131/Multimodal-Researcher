"""LangGraph implementation of the research and report generation workflow"""

from google.genai import types
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from agent.configuration import Configuration
from agent.state import ResearchState, ResearchStateInput, ResearchStateOutput
from agent.utils import display_gemini_response, genai_client


def web_search_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that performs web search research on the topic"""
    configuration = Configuration.from_runnable_config(config)

    if not state.topic and not state.video_url:
        raise ValueError("Either topic or video URL is required for search research")

    search_response = genai_client.models.generate_content(
        model=configuration.search_model,
        contents=f"Research this topic and give me an overview: {state.topic}",
        config={
            "tools": [{"google_search": {}}],
            "temperature": configuration.search_temperature,
        },
    )

    search_text, search_sources_text = display_gemini_response(search_response)

    return {
        "search_text": search_text,
        "search_sources_text": search_sources_text,
    }


def analyze_video_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that analyzes video content if video URL is provided"""
    configuration = Configuration.from_runnable_config(config)

    if not state.video_url:
        return {"video_text": "No video provided for analysis."}

    video_response = genai_client.models.generate_content(
        model=configuration.video_model,
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=state.video_url)),
                types.Part(text=f"Based on the video content, give me an overview of this topic: {state.topic}"),
            ]
        ),
    )

    video_text, _ = display_gemini_response(video_response)

    return {"video_text": video_text}


def create_report_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that creates a comprehensive research report"""
    configuration = Configuration.from_runnable_config(config)

    if not state.topic:
        raise ValueError("Topic is required for report creation")

    # Step 1: Create synthesis using Gemini
    synthesis_prompt = f"""
    You are a research analyst. I have gathered information about "{state.topic}" from two sources:
    
    SEARCH RESULTS:
    {state.search_text}
    
    VIDEO CONTENT:
    {state.video_text}
    
    Please create a comprehensive synthesis that:
    1. Identifies key themes and insights from both sources
    2. Highlights any complementary or contrasting perspectives
    3. Provides an overall analysis of the topic based on this multi-modal research
    4. Keep it concise but thorough (3-4 paragraphs)
    
    Focus on creating a coherent narrative that brings together the best insights from both sources.
    """

    synthesis_response = genai_client.models.generate_content(
        model=configuration.synthesis_model,
        contents=synthesis_prompt,
        config={
            "temperature": configuration.synthesis_temperature,
        },
    )

    synthesis_text = synthesis_response.candidates[0].content.parts[0].text

    # Step 2: Create markdown report
    report = f"""# Research Report: {state.topic}

## Executive Summary

{state.synthesis_text}

## Video Source
- **URL**: {state.video_url}

## Additional Sources
{state.search_sources_text}

---
*Report generated using multi-modal AI research combining web search and video analysis*
"""

    return {
        "report": report,
        "synthesis_text": synthesis_text,
    }


def should_analyze_video(state: ResearchState) -> str:
    """Conditional edge to determine if video analysis should be performed"""
    if state.video_url:
        return "analyze_video"
    else:
        return "create_report"


def create_research_graph() -> StateGraph:
    """Create and return the research workflow graph"""

    # Create the graph with configuration schema
    graph = StateGraph(
        ResearchState,
        input=ResearchStateInput,
        output=ResearchStateOutput,
        config_schema=Configuration,
    )

    # Add nodes
    graph.add_node("web_search", web_search_node)
    graph.add_node("analyze_video", analyze_video_node)
    graph.add_node("create_report", create_report_node)

    # Add edges
    graph.add_edge(START, "web_search")
    graph.add_conditional_edges("web_search", should_analyze_video, {"analyze_video": "analyze_video", "create_report": "create_report"})
    graph.add_edge("analyze_video", "create_report")
    graph.add_edge("create_report", END)

    return graph


def create_compiled_graph():
    """Create and compile the research graph"""
    graph = create_research_graph()
    return graph.compile()
