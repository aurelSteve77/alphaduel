"""alphaduel — controlled duel between LLM agents and quant models."""

from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the project-root .env before anything else so
# secrets (e.g. FINNHUB_API_KEY) are available to all modules on import.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from alphaduel.environments import AlphaDuelEnv, TradingEnv, TradingEnvConfig  # noqa: E402
from alphaduel.logger import get_logger, setup_logging  # noqa: E402
from alphaduel.strategies import LLMPolicy, LLMPolicyConfig  # noqa: E402

setup_logging()

__all__ = [
    "AlphaDuelEnv",
    "LLMPolicy",
    "LLMPolicyConfig",
    "TradingEnv",
    "TradingEnvConfig",
    "get_logger",
    "setup_logging",
]
