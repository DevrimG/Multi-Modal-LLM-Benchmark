"""Multi-Modal LLM Benchmark package."""

__version__ = "1.0.0"
__author__ = "Multi-Modal LLM Benchmark"

from .benchmarker import LLMBenchmarker, LoadTestConfig
from .metrics import BenchmarkResult, RequestMetrics, ErrorCategory
from .modalities import (
    ModalityHandler,
    TextHandler,
    ImageHandler,
    VoiceHandler,
    get_handler
)

__all__ = [
    "LLMBenchmarker",
    "LoadTestConfig",
    "BenchmarkResult",
    "RequestMetrics",
    "ErrorCategory",
    "ModalityHandler",
    "TextHandler",
    "ImageHandler",
    "VoiceHandler",
    "get_handler",
]
