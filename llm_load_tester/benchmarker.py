"""
Core benchmark engine for asynchronous LLM load testing.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.panel import Panel
from rich.live import Live

from .metrics import BenchmarkResult, ErrorCategory, RequestMetrics
from .modalities import ModalityHandler, PayloadResult


@dataclass
class LoadTestConfig:
    """Configuration for a load test run."""
    endpoint: str
    api_route: str
    model: str
    concurrency: int
    target_rps: float
    total_requests: int
    warmup_requests: int
    modality_handler: ModalityHandler
    modality_config: dict[str, Any]
    timeout_seconds: float = 120.0


class AsyncRateLimiter:
    """Token bucket rate limiter for controlling RPS."""
    
    def __init__(self, target_rps: float):
        self.target_rps = max(0.0, target_rps)
        self.unlimited = self.target_rps == 0
        self.tokens = self.target_rps
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary to maintain target RPS."""
        if self.unlimited:
            return

        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.target_rps, self.tokens + elapsed * self.target_rps)
            self.last_update = now
            
            if self.tokens < 1:
                # Need to wait for a token
                wait_time = (1 - self.tokens) / self.target_rps
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class LLMBenchmarker:
    """Main benchmark engine for LLM API load testing."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.console = Console()
        self.result = BenchmarkResult(
            modality=config.modality_config.get("modality", "unknown"),
            model=config.model,
            endpoint=f"{config.endpoint}/{config.api_route}",
            concurrency=config.concurrency,
            target_rps=config.target_rps,
            total_requests=config.total_requests,
            warmup_requests=config.warmup_requests,
            start_time=datetime.now(),
            # Text modality config
            input_tokens=config.modality_config.get("input_tokens"),
            output_tokens=config.modality_config.get("output_tokens"),
            # Image modality config
            image_directory=config.modality_config.get("image_directory"),
            # Voice modality config
            audio_directory=config.modality_config.get("audio_directory"),
            audio_file=config.modality_config.get("audio_file")
        )
        self.rate_limiter = AsyncRateLimiter(config.target_rps)
        self.request_counter = 0
        self.counter_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(config.concurrency)
        
    async def _get_next_request_id(self) -> int:
        """Get the next request ID atomically."""
        async with self.counter_lock:
            self.request_counter += 1
            return self.request_counter
    
    def _parse_sse_line(self, line: str) -> dict[str, Any] | None:
        """Parse a Server-Sent Events line."""
        if line.startswith("data: "):
            data = line[6:]  # Remove "data: " prefix
            if data == "[DONE]":
                return None
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        return None
    
    def _count_tokens_from_chunk(self, chunk: dict[str, Any]) -> int:
        """Count tokens from a streaming chunk."""
        count = 0
        choices = chunk.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            # Rough estimation: ~4 characters per token
            if content:
                count += max(1, len(content) // 4)
            # Also check for finish_reason
            if choice.get("finish_reason"):
                break
        return count
    
    def _extract_content_from_chunk(self, chunk: dict[str, Any]) -> str:
        """Extract content text from a streaming chunk."""
        content_parts = []
        choices = chunk.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            if content:
                content_parts.append(content)
        return "".join(content_parts)
    
    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        is_warmup: bool = False
    ) -> RequestMetrics:
        """Execute a single request and collect metrics."""
        request_id = await self._get_next_request_id()
        url = f"{self.config.endpoint}/{self.config.api_route}"
        
        # Generate a fresh unique payload for this request
        payload_result = await self.config.modality_handler.prepare_payload(
            self.config.modality_config
        )
        
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=time.monotonic(),
            input_tokens=payload_result.metadata.get("input_tokens", 0)
        )
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            async with session.post(
                url,
                json=payload_result.payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            ) as response:
                metrics.status_code = response.status
                
                if response.status != 200:
                    metrics.error = ErrorCategory.from_status_code(response.status)
                    error_text = await response.text()
                    metrics.error_message = error_text[:200]
                    metrics.end_time = time.monotonic()
                    return metrics
                
                # Process streaming response
                tokens_generated = 0
                first_token_received = False
                response_content_parts = []
                
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    
                    if not line:
                        continue
                    
                    chunk = self._parse_sse_line(line)
                    if chunk is None:
                        continue
                    
                    # Record TTFT on first chunk with content
                    if not first_token_received:
                        metrics.first_token_time = time.monotonic()
                        first_token_received = True
                    
                    # Count tokens from this chunk
                    tokens_generated += self._count_tokens_from_chunk(chunk)
                    
                    # Collect response content
                    content = self._extract_content_from_chunk(chunk)
                    if content:
                        response_content_parts.append(content)
                
                metrics.tokens_generated = tokens_generated
                metrics.response_content = "".join(response_content_parts)
                metrics.end_time = time.monotonic()
                
        except asyncio.TimeoutError:
            metrics.error = ErrorCategory.TIMEOUT
            metrics.error_message = "Request timed out"
            metrics.end_time = time.monotonic()
        except aiohttp.ClientError as e:
            metrics.error = ErrorCategory.CONNECTION_ERROR
            metrics.error_message = str(e)[:200]
            metrics.end_time = time.monotonic()
        except Exception as e:
            metrics.error = ErrorCategory.UNKNOWN
            metrics.error_message = str(e)[:200]
            metrics.end_time = time.monotonic()
        
        return metrics
    
    async def _execute_warmup(self, session: aiohttp.ClientSession) -> None:
        """Execute warmup requests to initialize the model."""
        self.console.print()
        self.console.print(Panel(
            f"[yellow]Warm-up Phase: Sending {self.config.warmup_requests} requests to initialize model...[/yellow]",
            title="Warm-up",
            border_style="yellow"
        ))
        
        warmup_tasks = []
        for _ in range(self.config.warmup_requests):
            # Each warmup request gets its own unique payload
            task = self._make_request(session, is_warmup=True)
            warmup_tasks.append(task)
        
        # Execute warmup requests
        completed = 0
        with Progress() as progress:
            task = progress.add_task("[yellow]Warming up...", total=self.config.warmup_requests)
            
            for coro in asyncio.as_completed(warmup_tasks):
                await coro
                completed += 1
                progress.update(task, advance=1)
        
        self.console.print("[green]✓ Warm-up complete![/green]")
        await asyncio.sleep(1)  # Brief pause after warmup
    
    async def _worker(
        self,
        session: aiohttp.ClientSession,
        progress: Progress,
        progress_task: TaskID
    ) -> None:
        """Worker coroutine that runs requests with rate limiting."""
        while True:
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Check if we've completed enough requests
            async with self.counter_lock:
                if self.request_counter >= self.config.total_requests:
                    break
            
            # Execute request with concurrency control (each gets unique payload)
            async with self.semaphore:
                metrics = await self._make_request(session)
                self.result.add_request(metrics)
                progress.update(progress_task, advance=1)
    
    async def run(self) -> BenchmarkResult:
        """Execute the full benchmark."""
        url = f"{self.config.endpoint}/{self.config.api_route}"
        self.console.print(Panel(
            f"[cyan]Target: {url}\n"
            f"Model: {self.config.model}\n"
            f"Concurrency: {self.config.concurrency}\n"
            f"Target RPS: {'Unlimited' if self.config.target_rps <= 0 else self.config.target_rps}\n"
            f"Total Requests: {self.config.total_requests}[/cyan]",
            title="Benchmark Configuration",
            border_style="cyan"
        ))
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrency * 2,
            limit_per_host=self.config.concurrency * 2,
            enable_cleanup_closed=True,
            force_close=True
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # Warmup phase
            if self.config.warmup_requests > 0:
                await self._execute_warmup(session)
            
            # Reset counter for actual benchmark
            self.request_counter = 0
            
            self.console.print()
            self.console.print(Panel(
                "[green]Starting benchmark...[/green]",
                title="Benchmark",
                border_style="green"
            ))
            
            # Main benchmark with progress tracking
            with Progress() as progress:
                task = progress.add_task(
                    "[green]Executing requests...",
                    total=self.config.total_requests
                )
                
                # Create worker tasks
                workers = [
                    self._worker(session, progress, task)
                    for _ in range(self.config.concurrency)
                ]
                
                # Run all workers
                await asyncio.gather(*workers)
        
        # Record end time
        self.result.end_time = datetime.now()
        
        return self.result
