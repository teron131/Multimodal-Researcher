"""Configuration settings for the research and report generation app"""

import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """LangGraph Configuration for the deep research agent."""

    # Model settings
    plan_model: str = Field(default="gemini-2.5-flash", description="Plan supported model")
    search_model: str = Field(default="gemini-2.5-flash", description="Web search supported model")
    synthesis_model: str = Field(default="gemini-2.5-flash", description="Citations supported model")
    video_model: str = Field(default="gemini-2.5-flash", description="Video analysis supported model")

    # Temperature settings for different use cases
    plan_temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Plan temperature")
    search_temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Factual search queries")
    synthesis_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Balanced synthesis")

    class Config:
        env_prefix = ""
        case_sensitive = False

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        if not config or "configurable" not in config:
            return cls()

        configurable = config["configurable"]

        # Get field names from the model
        field_names = cls.model_fields.keys()

        # Build values dict from environment variables and configurable
        values: dict[str, Any] = {}
        for field_name in field_names:
            env_value = os.environ.get(field_name.upper())
            config_value = configurable.get(field_name)

            if env_value is not None:
                values[field_name] = env_value
            elif config_value is not None:
                values[field_name] = config_value

        return cls(**values)
