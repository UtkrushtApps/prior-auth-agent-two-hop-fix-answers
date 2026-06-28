"""Configuration loading from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = BASE_DIR / "fixtures"


@dataclass
class AppConfig:
    model: str
    api_key: str | None
    max_steps: int
    request_timeout: int

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def _load_config() -> AppConfig:
    return AppConfig(
        model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "5")),
        request_timeout=int(os.getenv("AGENT_REQUEST_TIMEOUT", "30")),
    )


config = _load_config()
