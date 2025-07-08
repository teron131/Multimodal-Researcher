"""LangGraph implementation of the research and report generation workflow"""

import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from google.genai import Client, types
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pytubefix import YouTube
from tqdm import tqdm

from multimodal_researcher.configuration import Configuration
from multimodal_researcher.state import (
    GraphInput,
    GraphOutput,
    GraphState,
    Plan,
    SearchResult,
    SearchResults,
    Section,
    VideoResult,
    VideoResults,
)
from multimodal_researcher.utils import extract_youtube_video_urls

load_dotenv()


client = Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"timeout": 600000},  # 10 minutes
)
google_search_tool = types.Tool(google_search=types.GoogleSearch())


def plan_node(state: GraphState, config: RunnableConfig) -> dict:
    """Node that plans the research"""
    configuration = Configuration.from_runnable_config(config)

    plan_response = client.models.generate_content(
        model=configuration.plan_model,
        contents=f"Plan the subtopics / questions to research for the topic: {state.topic} as a list of 3-5 sections",
        config=types.GenerateContentConfig(
            temperature=configuration.plan_temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
            response_mime_type="application/json",
            response_schema=Plan,
        ),
    )

    plan: Plan = plan_response.parsed

    return {"plan": plan}


def _web_search(section: Section, topic: str, configuration: Configuration) -> SearchResult:
    """Helper function to search a single section"""
    search_response = client.models.generate_content(
        model=configuration.search_model,
        contents=f"Research about the topic{topic} in {section.title}: draw key points and conclusions about {section.description}",
        config=types.GenerateContentConfig(
            tools=[google_search_tool],
            temperature=configuration.search_temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
        ),
    )

    answer = search_response.text
    # Parse the sources as a list
    sources = []
    for grounding_chunk in search_response.candidates[0].grounding_metadata.grounding_chunks:
        sources.append((grounding_chunk.web.uri, grounding_chunk.web.title))

    return SearchResult(section=section, answer=answer, sources=sources)


def web_search_node(state: GraphState, config: RunnableConfig) -> dict:
    """Node that performs web search research on the topic"""
    configuration = Configuration.from_runnable_config(config)

    if not state.topic and not state.video_urls_raw:
        raise ValueError("Either topic or video URL is required for search research")

    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(state.plan.sections))) as executor:
        search_tasks = [executor.submit(_web_search, section, state.topic, configuration) for section in state.plan.sections]
        search_results = list(tqdm([task.result() for task in search_tasks], total=len(search_tasks), desc="Searching sections"))

    return {"search_results": SearchResults(section_results=search_results)}


def _analyze_video(video_url: str, configuration: Configuration) -> VideoResult:
    """Helper function to analyze a single video"""

    video_response = client.models.generate_content(
        model=configuration.video_model,
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=video_url)),
                types.Part(text=f"As you watch the video, create a detailed note of the video as if an article, and give a summary."),
            ]
        ),
        config=types.GenerateContentConfig(
            temperature=configuration.video_temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=VideoResult,
        ),
    )

    video_result: VideoResult = video_response.parsed
    video_result.video_url = video_url
    video_result.video_title = YouTube(video_url).title

    return video_result


def analyze_video_node(state: GraphState, config: RunnableConfig) -> dict:
    """Node that analyzes video content if video URL is provided"""
    configuration = Configuration.from_runnable_config(config)

    if not state.video_urls_raw:
        return {"video_text": "No video provided for analysis."}

    video_urls = extract_youtube_video_urls(state.video_urls_raw)

    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(video_urls))) as executor:
        video_tasks = [executor.submit(_analyze_video, video_url, configuration) for video_url in video_urls]
        video_results = list(tqdm([task.result() for task in video_tasks], total=len(video_tasks), desc="Analyzing videos"))

    return {"video_results": VideoResults(video_results=video_results)}


def create_report_node(state: GraphState, config: RunnableConfig) -> dict:
    """Node that creates a comprehensive research report"""
    configuration = Configuration.from_runnable_config(config)

    if not state.topic:
        raise ValueError("Topic is required for report creation")

    # Step 1: Create synthesis using Gemini
    synthesis_prompt = f"""You are a research analyst. I have gathered information about "{state.topic}" from two sources:

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

    synthesis_response = client.models.generate_content(
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

{synthesis_text}

## Video Source
- **URL**: {state.video_url}

## Additional Sources
{state.search_sources_text}

---
*Report generated using multi-modal AI research combining web search and video analysis*
"""

    return {
        "output": {
            "report": report,
            "synthesis_text": synthesis_text,
        },
    }


def should_analyze_video(state: GraphState) -> str:
    """Conditional edge to determine if video analysis should be performed"""
    if state.video_url:
        return "analyze_video"
    else:
        return "create_report"


def create_research_graph() -> StateGraph:
    """Create and return the research workflow graph"""

    # Create the graph with configuration schema
    graph = StateGraph(
        GraphState,
        input=GraphInput,
        output=GraphOutput,
        config_schema=Configuration,
    )

    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("analyze_video", analyze_video_node)
    graph.add_node("create_report", create_report_node)

    # Add edges
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "web_search")
    graph.add_conditional_edges("web_search", should_analyze_video, {"analyze_video": "analyze_video", "create_report": "create_report"})
    graph.add_edge("analyze_video", "create_report")
    graph.add_edge("create_report", END)

    return graph


def create_compiled_graph():
    """Create and compile the research graph"""
    graph = create_research_graph()
    return graph.compile()
