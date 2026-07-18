"""alphaduel — a controlled duel between reasoning LLM agents and pure quant
models for portfolio allocation.

This package exposes a shared, framework-agnostic portfolio simulator and a
Gymnasium environment on top of it. Both the LLM branch and the quant branch
train and are evaluated against *this same* environment to guarantee parity.
"""

from .config import Config, load_config
from .data import MarketData
from .env import AlphaDuelEnv, PortfolioSimulator

__version__ = "0.0.1"

__all__ = [
    "Config",
    "load_config",
    "MarketData",
    "AlphaDuelEnv",
    "PortfolioSimulator",
    "__version__",
]
