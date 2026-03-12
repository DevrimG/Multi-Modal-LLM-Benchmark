"""
Metrics collection, calculation, and export for LLM load testing.
"""

import csv
import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from tabulate import tabulate


class ErrorCategory(Enum):
    """Categories of errors that can occur during testing."""
    NONE = "none"
    RATE_LIMIT = "rate_limit"  # 429
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    SERVER_ERROR = "server_error"  # 5xx
    CLIENT_ERROR = "client_error"  # 4xx
    UNKNOWN = "unknown"
    
    @classmethod
    def from_status_code(cls, status_code: int | None) -> "ErrorCategory":
        """Categorize error based on HTTP status code."""
        if status_code is None:
            return cls.CONNECTION_ERROR
        if status_code == 429:
            return cls.RATE_LIMIT
        if 500 <= status_code < 600:
            return cls.SERVER_ERROR
        if 400 <= status_code < 500:
            return cls.CLIENT_ERROR
        return cls.UNKNOWN


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: int
    start_time: float
    first_token_time: float | None = None
    end_time: float | None = None
    tokens_generated: int = 0
    input_tokens: int = 0
    error: ErrorCategory = ErrorCategory.NONE
    error_message: str = ""
    status_code: int | None = None
    response_content: str = ""
    
    @property
    def ttft(self) -> float | None:
        """Time To First Token in seconds."""
        if self.first_token_time and self.start_time:
            return self.first_token_time - self.start_time
        return None
    
    @property
    def total_latency(self) -> float | None:
        """Total end-to-end latency in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def tpot(self) -> float | None:
        """Time Per Output Token in seconds."""
        if self.tokens_generated > 0 and self.total_latency is not None:
            # Exclude TTFT from TPOT calculation
            if self.ttft is not None:
                return (self.total_latency - self.ttft) / self.tokens_generated
            return self.total_latency / self.tokens_generated
        return None
    
    @property
    def tokens_per_second(self) -> float | None:
        """Tokens per second for this request."""
        if self.tokens_generated > 0 and self.total_latency is not None:
            return self.tokens_generated / self.total_latency
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "request_id": self.request_id,
            "start_time": self.start_time,
            "first_token_time": self.first_token_time,
            "end_time": self.end_time,
            "ttft_seconds": self.ttft,
            "total_latency_seconds": self.total_latency,
            "tpot_seconds": self.tpot,
            "tokens_generated": self.tokens_generated,
            "input_tokens": self.input_tokens,
            "tokens_per_second": self.tokens_per_second,
            "error": self.error.value,
            "error_message": self.error_message,
            "status_code": self.status_code,
            "response_content": self.response_content
        }


@dataclass
class BenchmarkResult:
    """Aggregated results from a benchmark run."""
    # Configuration
    modality: str
    model: str
    endpoint: str
    concurrency: int
    target_rps: float
    total_requests: int
    warmup_requests: int
    
    # Timing
    start_time: datetime
    end_time: datetime | None = None
    
    # Text modality configuration
    input_tokens: int | None = None
    output_tokens: int | None = None
    
    # Image modality configuration
    image_directory: str | None = None
    
    # Voice modality configuration
    audio_directory: str | None = None
    audio_file: str | None = None
    
    # Raw metrics
    request_metrics: list[RequestMetrics] = field(default_factory=list)
    
    # Error tracking
    errors: dict[ErrorCategory, int] = field(default_factory=lambda: {
        cat: 0 for cat in ErrorCategory
    })
    
    def add_request(self, metrics: RequestMetrics) -> None:
        """Add a request's metrics to the results."""
        self.request_metrics.append(metrics)
        if metrics.error != ErrorCategory.NONE:
            self.errors[metrics.error] += 1
    
    @property
    def successful_requests(self) -> int:
        """Count of successful requests."""
        return sum(1 for m in self.request_metrics if m.error == ErrorCategory.NONE)
    
    @property
    def failed_requests(self) -> int:
        """Count of failed requests."""
        return sum(1 for m in self.request_metrics if m.error != ErrorCategory.NONE)
    
    @property
    def error_rate(self) -> float:
        """Error rate as a percentage."""
        if not self.request_metrics:
            return 0.0
        return (self.failed_requests / len(self.request_metrics)) * 100
    
    def _calculate_percentile(self, values: list[float], percentile: float) -> float | None:
        """Calculate percentile value from a list of numbers."""
        if not values:
            return None
        return np.percentile(values, percentile)
    
    def _get_valid_values(self, extractor) -> list[float]:
        """Extract valid (non-None) values from request metrics."""
        return [v for m in self.request_metrics if (v := extractor(m)) is not None]
    
    def get_summary(self) -> dict[str, Any]:
        """Generate a summary of benchmark results."""
        # Collect all valid values
        ttfts = self._get_valid_values(lambda m: m.ttft)
        latencies = self._get_valid_values(lambda m: m.total_latency)
        tpots = self._get_valid_values(lambda m: m.tpot)
        tps_per_req = self._get_valid_values(lambda m: m.tokens_per_second)
        
        # Total tokens
        total_output_tokens = sum(m.tokens_generated for m in self.request_metrics)
        total_input_tokens = sum(m.input_tokens for m in self.request_metrics)
        
        # Overall duration
        duration_seconds = None
        if self.end_time and self.start_time:
            duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        # Overall throughput
        overall_tps = None
        if duration_seconds and duration_seconds > 0:
            overall_tps = total_output_tokens / duration_seconds
        
        summary = {
            # Configuration
            "modality": self.modality,
            "model": self.model,
            "endpoint": self.endpoint,
            "concurrency": self.concurrency,
            "target_rps": self.target_rps,
            "total_requests": self.total_requests,
            "warmup_requests": self.warmup_requests,
            "actual_requests": len(self.request_metrics),
            
            # Modality-specific configuration
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "image_directory": self.image_directory,
            "audio_directory": self.audio_directory,
            "audio_file": self.audio_file,
            
            # Timing
            "duration_seconds": duration_seconds,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            
            # Success metrics
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate_percent": round(self.error_rate, 2),
            
            # Token counts
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            
            # TTFT (Time To First Token)
            "ttft_mean": round(statistics.mean(ttfts), 3) if ttfts else None,
            "ttft_p50": round(statistics.median(ttfts), 3) if ttfts else None,
            "ttft_p95": round(self._calculate_percentile(ttfts, 95), 3) if ttfts else None,
            "ttft_p99": round(self._calculate_percentile(ttfts, 99), 3) if ttfts else None,
            "ttft_min": round(min(ttfts), 3) if ttfts else None,
            "ttft_max": round(max(ttfts), 3) if ttfts else None,
            
            # End-to-End Latency
            "latency_mean": round(statistics.mean(latencies), 3) if latencies else None,
            "latency_p50": round(statistics.median(latencies), 3) if latencies else None,
            "latency_p95": round(self._calculate_percentile(latencies, 95), 3) if latencies else None,
            "latency_p99": round(self._calculate_percentile(latencies, 99), 3) if latencies else None,
            "latency_min": round(min(latencies), 3) if latencies else None,
            "latency_max": round(max(latencies), 3) if latencies else None,
            
            # TPOT (Time Per Output Token)
            "tpot_mean": round(statistics.mean(tpots), 4) if tpots else None,
            "tpot_p50": round(statistics.median(tpots), 4) if tpots else None,
            "tpot_p95": round(self._calculate_percentile(tpots, 95), 4) if tpots else None,
            "tpot_p99": round(self._calculate_percentile(tpots, 99), 4) if tpots else None,
            
            # Throughput
            "overall_tokens_per_second": round(overall_tps, 2) if overall_tps else None,
            "per_request_tokens_per_second_mean": round(statistics.mean(tps_per_req), 2) if tps_per_req else None,
            "per_request_tokens_per_second_p50": round(statistics.median(tps_per_req), 2) if tps_per_req else None,
            
            # Errors
            "errors_by_category": {
                cat.value: count for cat, count in self.errors.items() if count > 0
            }
        }
        
        return summary
    
    def print_rich_table(self) -> None:
        """Print results using Rich for formatted terminal output."""
        console = Console()
        summary = self.get_summary()
        
        # Main results table
        table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Configuration section
        table.add_row("[bold]Configuration[/bold]", "")
        table.add_row("  Modality", summary["modality"])
        table.add_row("  Model", summary["model"])
        table.add_row("  Endpoint", summary["endpoint"])
        table.add_row("  Concurrency", str(summary["concurrency"]))
        table.add_row("  Target RPS", str(summary["target_rps"]))
        table.add_row("  Total Requests", str(summary["total_requests"]))
        table.add_row("  Warmup Requests", str(summary["warmup_requests"]))
        
        # Modality-specific configuration
        if summary["modality"] == "text":
            if summary["input_tokens"]:
                table.add_row("  Input Tokens", str(summary["input_tokens"]))
            if summary["output_tokens"]:
                table.add_row("  Output Tokens", str(summary["output_tokens"]))
        elif summary["modality"] == "image" and summary["image_directory"]:
            table.add_row("  Image Directory", summary["image_directory"])
        elif summary["modality"] == "voice":
            if summary["audio_directory"]:
                table.add_row("  Sound Directory", summary["audio_directory"])
            elif summary["audio_file"]:
                table.add_row("  Audio File", summary["audio_file"])
        
        table.add_row("", "")
        
        # Summary section
        table.add_row("[bold]Summary[/bold]", "")
        table.add_row("  Duration", f"{summary['duration_seconds']:.2f}s" if summary['duration_seconds'] else "N/A")
        table.add_row("  Successful Requests", str(summary["successful_requests"]))
        table.add_row("  Failed Requests", str(summary["failed_requests"]))
        table.add_row("  Error Rate", f"{summary['error_rate_percent']:.2f}%")
        table.add_row("  Total Output Tokens", str(summary["total_output_tokens"]))
        table.add_row("  Overall Throughput", f"{summary['overall_tokens_per_second']:.2f} tok/s" if summary["overall_tokens_per_second"] else "N/A")
        table.add_row("", "")
        
        # TTFT section
        table.add_row("[bold]Time To First Token (TTFT)[/bold]", "")
        table.add_row("  Mean", f"{summary['ttft_mean']:.3f}s" if summary["ttft_mean"] else "N/A")
        table.add_row("  p50", f"{summary['ttft_p50']:.3f}s" if summary["ttft_p50"] else "N/A")
        table.add_row("  p95", f"{summary['ttft_p95']:.3f}s" if summary["ttft_p95"] else "N/A")
        table.add_row("  p99", f"{summary['ttft_p99']:.3f}s" if summary["ttft_p99"] else "N/A")
        table.add_row("  Range", f"{summary['ttft_min']:.3f}s - {summary['ttft_max']:.3f}s" if summary["ttft_min"] else "N/A")
        table.add_row("", "")
        
        # Latency section
        table.add_row("[bold]End-to-End Latency[/bold]", "")
        table.add_row("  Mean", f"{summary['latency_mean']:.3f}s" if summary["latency_mean"] else "N/A")
        table.add_row("  p50", f"{summary['latency_p50']:.3f}s" if summary["latency_p50"] else "N/A")
        table.add_row("  p95", f"{summary['latency_p95']:.3f}s" if summary["latency_p95"] else "N/A")
        table.add_row("  p99", f"{summary['latency_p99']:.3f}s" if summary["latency_p99"] else "N/A")
        table.add_row("  Range", f"{summary['latency_min']:.3f}s - {summary['latency_max']:.3f}s" if summary["latency_min"] else "N/A")
        table.add_row("", "")
        
        # TPOT section
        table.add_row("[bold]Time Per Output Token (TPOT)[/bold]", "")
        table.add_row("  Mean", f"{summary['tpot_mean']:.4f}s" if summary["tpot_mean"] else "N/A")
        table.add_row("  p50", f"{summary['tpot_p50']:.4f}s" if summary["tpot_p50"] else "N/A")
        table.add_row("  p95", f"{summary['tpot_p95']:.4f}s" if summary["tpot_p95"] else "N/A")
        table.add_row("  p99", f"{summary['tpot_p99']:.4f}s" if summary["tpot_p99"] else "N/A")
        table.add_row("", "")
        
        # Throughput section
        table.add_row("[bold]Throughput[/bold]", "")
        table.add_row("  Overall", f"{summary['overall_tokens_per_second']:.2f} tok/s" if summary["overall_tokens_per_second"] else "N/A")
        table.add_row("  Per Request Mean", f"{summary['per_request_tokens_per_second_mean']:.2f} tok/s" if summary["per_request_tokens_per_second_mean"] else "N/A")
        table.add_row("  Per Request p50", f"{summary['per_request_tokens_per_second_p50']:.2f} tok/s" if summary["per_request_tokens_per_second_p50"] else "N/A")
        
        console.print()
        console.print(table)
        
        # Error breakdown if any
        if summary["errors_by_category"]:
            error_table = Table(title="Error Breakdown", show_header=True, header_style="bold red")
            error_table.add_column("Error Category", style="red")
            error_table.add_column("Count", style="yellow")
            for cat, count in summary["errors_by_category"].items():
                error_table.add_row(cat, str(count))
            console.print()
            console.print(error_table)
        
        console.print()
    
    def export_json(self, filepath: Path) -> None:
        """Export results to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        export_data = {
            "summary": self.get_summary(),
            "raw_metrics": [m.to_dict() for m in self.request_metrics]
        }
        
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)
    
    def export_csv(self, filepath: Path) -> None:
        """Export raw metrics to CSV file."""
        if not self.request_metrics:
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = list(self.request_metrics[0].to_dict().keys())
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metric in self.request_metrics:
                writer.writerow(metric.to_dict())
