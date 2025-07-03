"""LangGraph implementation of the research and report generation workflow"""

from google.genai import types
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from agent.configuration import Configuration
from agent.state import ResearchState, ResearchStateInput, ResearchStateOutput
from agent.utils import create_research_report, display_gemini_response, genai_client


@traceable(run_type="llm", name="Web Research", project_name="multimodal-researcher")
def search_research_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that performs web search research on the topic"""
    configuration = Configuration.from_runnable_config(config)
    topic = state.topic

    if not topic:
        raise ValueError("Topic is required for search research")

    search_response = genai_client.models.generate_content(
        model=configuration.search_model,
        contents=f"Research this topic and give me an overview: {topic}",
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


@traceable(run_type="llm", name="YouTube Video Analysis", project_name="multimodal-researcher")
def analyze_video_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that analyzes video content if video URL is provided"""
    configuration = Configuration.from_runnable_config(config)
    video_url = state.video_url
    topic = state.topic

    if not topic:
        raise ValueError("Topic is required for video analysis")

    if not video_url:
        return {"video_text": "No video provided for analysis."}

    video_response = genai_client.models.generate_content(
        model=configuration.video_model,
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=video_url)),
                types.Part(text=f"Based on the video content, give me an overview of this topic: {topic}"),
            ]
        ),
    )

    video_text, _ = display_gemini_response(video_response)

    return {"video_text": video_text}


@traceable(run_type="llm", name="Create Report", project_name="multimodal-researcher")
def create_report_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that creates a comprehensive research report"""
    configuration = Configuration.from_runnable_config(config)
    topic = state.topic
    search_text = state.search_text or ""
    video_text = state.video_text or ""
    search_sources_text = state.search_sources_text or ""
    video_url = state.video_url or ""

    if not topic:
        raise ValueError("Topic is required for report creation")

    report, synthesis_text = create_research_report(
        topic,
        search_text,
        video_text,
        search_sources_text,
        video_url,
        configuration,
    )

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
    graph.add_node("search_research", search_research_node)
    graph.add_node("analyze_video", analyze_video_node)
    graph.add_node("create_report", create_report_node)

    # Add edges
    graph.add_edge(START, "search_research")
    graph.add_conditional_edges("search_research", should_analyze_video, {"analyze_video": "analyze_video", "create_report": "create_report"})
    graph.add_edge("analyze_video", "create_report")
    graph.add_edge("create_report", END)

    return graph


def create_compiled_graph():
    """Create and compile the research graph"""
    graph = create_research_graph()
    return graph.compile()
