"""Interactive CLI interface for the benchmark tool."""

import asyncio
from pathlib import Path
from typing import Any

import aiohttp
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm

from .modalities import get_handler


console = Console()


# Preset configurations
API_PRESETS = {
    "OpenAI Compatible (Local)": {
        "endpoint": "http://localhost:8000",
        "api_route": "v1/chat/completions"
    },
    "vLLM Default": {
        "endpoint": "http://localhost:8000",
        "api_route": "v1/chat/completions"
    },
    "Ollama": {
        "endpoint": "http://localhost:11434",
        "api_route": "v1/chat/completions"
    },
    "Custom": None
}

TOKEN_PRESETS = [256, 512, 1024, 2048, 4096, 8192]

CONCURRENCY_PRESETS = [1, 2, 4, 8, 16, 32, 64]

RPS_PRESETS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

REQUEST_COUNT_PRESETS = [10, 50, 100, 200, 500, 1000]


def print_header() -> None:
    """Print the application header."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Multi-Modal LLM Benchmark[/bold cyan]\n"
        "[dim]Asynchronous benchmarking tool for LLM APIs[/dim]",
        border_style="cyan"
    ))
    console.print()


def select_modality() -> str:
    """Prompt user to select test modality."""
    console.print("[bold]Select Test Modality:[/bold]")
    console.print("  1. Text - Text-based chat completions")
    console.print("  2. Image - Vision/multimodal with images")
    console.print("  3. Voice - Audio/voice inputs")
    console.print()
    
    valid_choices = {"1", "2", "3", "text", "image", "voice"}
    mapping = {"1": "text", "2": "image", "3": "voice"}
    
    while True:
        choice = Prompt.ask("Enter choice (1)", default="1").lower()
        if choice in valid_choices:
            return mapping.get(choice, choice)
        console.print("[red]Invalid choice. Please enter 1, 2, 3, text, image, or voice.[/red]")


def select_from_presets(
    title: str,
    presets: list[str] | dict[str, Any],
    allow_custom: bool = True,
    custom_prompt: str = "Enter custom value"
) -> str | dict[str, Any] | None:
    """Generic preset selector with custom option."""
    console.print(f"\n[bold]{title}[/bold]")
    
    if isinstance(presets, dict):
        items = list(presets.keys())
    else:
        items = list(presets)
    
    for i, item in enumerate(items, 1):
        if isinstance(presets, dict) and presets[item] is not None:
            details = presets[item]
            if isinstance(details, dict):
                detail_str = f" ({details.get('endpoint', '')})"
            else:
                detail_str = ""
            console.print(f"  {i}. {item}{detail_str}")
        else:
            console.print(f"  {i}. {item}")
    
    console.print()
    
    # Build choices
    num_choices = [str(i) for i in range(1, len(items) + 1)]
    name_choices = [item.lower() for item in items]
    choices = num_choices + name_choices
    
    valid_choices = set(str(i) for i in range(1, len(items) + 1)) | set(item.lower() for item in items)
    
    while True:
        choice = Prompt.ask("Enter choice (1)", default="1").lower()
        if choice in valid_choices:
            break
        console.print(f"[red]Invalid choice. Please enter 1-{len(items)} or the option name.[/red]")
    
    # Convert numeric choice to item name
    if choice.isdigit():
        selected = items[int(choice) - 1]
    else:
        selected = choice
        # Find the actual case-sensitive item
        for item in items:
            if item.lower() == selected:
                selected = item
                break
    
    # Handle dict presets
    if isinstance(presets, dict):
        value = presets.get(selected)
        if value is None:  # Custom selected
            return None
        return selected, value
    
    # Handle list presets
    if selected.lower() == "custom":
        return Prompt.ask(custom_prompt)
    
    return selected


async def test_endpoint(endpoint: str, api_route: str) -> tuple[bool, list[str] | None, str]:
    """Test connection to endpoint and fetch available models.
    
    Returns: (success, available_models, message)
    """
    url = f"{endpoint}/{api_route}"
    models_url = f"{endpoint}/v1/models"
    
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Try to fetch models first - this validates the endpoint exists
            models = None
            try:
                async with session.get(models_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and isinstance(data["data"], list):
                            models = [m.get("id", m.get("name", "unknown")) for m in data["data"]]
                            return True, models, "Endpoint is reachable"
                    elif response.status in (401, 403):
                        return True, models, "Endpoint is reachable (authentication required for models list)"
            except Exception:
                pass  # Model endpoint might not exist
            
            # Test chat completions endpoint with a minimal request
            # Try with a common model name that's likely to exist
            test_models = ["gpt-3.5-turbo", "gpt-4", "default", "model"]
            last_error = ""
            
            for test_model in test_models:
                test_payload = {
                    "model": test_model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1
                }
                
                try:
                    async with session.post(url, json=test_payload) as response:
                        if response.status == 200:
                            return True, models, "Endpoint is reachable"
                        elif response.status in (401, 403):
                            return True, models, "Endpoint is reachable (authentication required)"
                        elif response.status == 404:
                            # Model not found - try next model
                            text = await response.text()
                            if "does not exist" in text.lower() or "not found" in text.lower():
                                last_error = f"Model '{test_model}' not found on server"
                                continue
                            return True, models, "Endpoint is reachable (model may need to be specified)"
                        elif response.status == 422:
                            return True, models, "Endpoint is reachable (validation error - check parameters)"
                        else:
                            text = await response.text()
                            return False, models, f"Endpoint returned status {response.status}: {text[:100]}"
                except Exception as e:
                    return False, models, f"Connection failed: {str(e)[:100]}"
            
            # If all model tests failed with 404, the endpoint exists but needs correct model name
            if last_error:
                return True, models, f"Endpoint is reachable (server requires a valid model name)"
            
            return False, models, "Could not validate endpoint"
                
    except Exception as e:
        return False, None, f"Connection failed: {str(e)[:100]}"


def select_api_config() -> tuple[str, str, list[str] | None]:
    """Prompt user for API endpoint and route configuration.
    
    Returns: (endpoint, api_route, available_models)
    """
    result = select_from_presets(
        "Select API Configuration Preset:",
        API_PRESETS,
        allow_custom=True
    )
    
    if result is None or isinstance(result, str):  # Custom selected
        console.print("\n[dim]Enter custom API configuration:[/dim]")
        endpoint = Prompt.ask(
            "Endpoint IP:Port",
            default="localhost:8000"
        )
        api_route = Prompt.ask(
            "API Route",
            default="v1/chat/completions"
        )
        # Ensure endpoint has http:// prefix
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"http://{endpoint}"
    else:
        name, config = result
        endpoint = config["endpoint"]
        api_route = config["api_route"]
    
    # Test the endpoint
    console.print(f"\n[yellow]Testing connection to {endpoint}...[/yellow]")
    success, models, message = asyncio.run(test_endpoint(endpoint, api_route))
    
    if success:
        console.print(f"[green]✓ {message}[/green]")
        if models:
            console.print(f"[green]✓ Found {len(models)} model(s)[/green]")
    else:
        console.print(f"[red]✗ {message}[/red]")
        console.print("[yellow]Warning: Endpoint test failed. Continuing anyway...[/yellow]")
    
    return endpoint, api_route, models


def select_model(available_models: list[str] | None = None) -> str:
    """Prompt user to enter model name."""
    console.print("\n[bold]Enter Model Name:[/bold]")
    
    if available_models:
        console.print("\n[dim]Available models on this endpoint:[/dim]")
        display_models = available_models[:20]  # Show first 20
        for i, m in enumerate(display_models, 1):
            console.print(f"  {i}. {m}")
        if len(available_models) > 20:
            console.print(f"  ... and {len(available_models) - 20} more")
        console.print()
        console.print("[dim]Enter a number to select from above, or type a custom model name[/dim]")
        
        while True:
            choice = Prompt.ask("Model name or number", default="1")
            # Check if user entered a number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_models):
                    return available_models[idx]
                else:
                    console.print(f"[red]Invalid number. Please enter 1-{len(available_models)} or a model name.[/red]")
            else:
                # User entered a custom model name
                return choice
    else:
        console.print("[dim]Examples: gpt-4, meta-llama/Llama-2-7b-chat-hf, mistral-7b-instruct[/dim]")
        console.print()
        return Prompt.ask("Model name", default="gpt-4")


def select_token_length(prompt_text: str) -> int:
    """Prompt user to select token length with presets."""
    console.print(f"\n[bold]{prompt_text}[/bold]")
    
    for i, preset in enumerate(TOKEN_PRESETS, 1):
        console.print(f"  {i}. {preset} tokens")
    console.print(f"  {len(TOKEN_PRESETS) + 1}. Custom")
    console.print()
    
    max_choice = len(TOKEN_PRESETS) + 1
    valid_choices = set(str(i) for i in range(1, max_choice + 1))
    
    while True:
        choice = Prompt.ask("Enter choice (1)", default="1")
        if choice in valid_choices:
            break
        console.print(f"[red]Invalid choice. Please enter 1-{max_choice}.[/red]")
    
    if int(choice) == len(TOKEN_PRESETS) + 1:
        return IntPrompt.ask("Enter custom token count", default=512)
    
    return TOKEN_PRESETS[int(choice) - 1]


def select_concurrency() -> int:
    """Prompt user for concurrency level."""
    console.print("\n[bold]Select Concurrency Level:[/bold]")
    console.print("[dim](Number of simultaneous connections)[/dim]")
    
    for i, preset in enumerate(CONCURRENCY_PRESETS, 1):
        console.print(f"  {i}. {preset} concurrent")
    console.print(f"  {len(CONCURRENCY_PRESETS) + 1}. Custom")
    console.print()
    
    max_choice = len(CONCURRENCY_PRESETS) + 1
    valid_choices = set(str(i) for i in range(1, max_choice + 1))
    
    while True:
        choice = Prompt.ask("Enter choice (4)", default="4")
        if choice in valid_choices:
            break
        console.print(f"[red]Invalid choice. Please enter 1-{max_choice}.[/red]")
    
    if int(choice) == len(CONCURRENCY_PRESETS) + 1:
        return IntPrompt.ask("Enter custom concurrency", default=4)
    
    return CONCURRENCY_PRESETS[int(choice) - 1]


def select_rps() -> float:
    """Prompt user for target RPS."""
    console.print("\n[bold]Select Target RPS:[/bold]")
    console.print("[dim](Requests Per Second - use 0 for unlimited)[/dim]")
    
    for i, preset in enumerate(RPS_PRESETS, 1):
        console.print(f"  {i}. {preset} RPS")
    console.print(f"  {len(RPS_PRESETS) + 1}. Custom")
    console.print(f"  {len(RPS_PRESETS) + 2}. Unlimited (0)")
    console.print()
    
    max_choice = len(RPS_PRESETS) + 2
    valid_choices = set(str(i) for i in range(1, max_choice + 1))
    
    while True:
        choice = Prompt.ask("Enter choice (5)", default="5")
        if choice in valid_choices:
            break
        console.print(f"[red]Invalid choice. Please enter 1-{max_choice}.[/red]")
    
    choice_int = int(choice)
    if choice_int == len(RPS_PRESETS) + 1:
        while True:
            custom_rps = FloatPrompt.ask("Enter custom RPS", default=1.0)
            if custom_rps >= 0:
                return custom_rps
            console.print("[red]RPS must be 0 or greater.[/red]")
    elif choice_int == len(RPS_PRESETS) + 2:
        return 0.0
    
    return RPS_PRESETS[choice_int - 1]


def select_request_count() -> int:
    """Prompt user for total request count."""
    console.print("\n[bold]Select Total Request Count:[/bold]")
    
    for i, preset in enumerate(REQUEST_COUNT_PRESETS, 1):
        console.print(f"  {i}. {preset} requests")
    console.print(f"  {len(REQUEST_COUNT_PRESETS) + 1}. Custom")
    console.print()
    
    max_choice = len(REQUEST_COUNT_PRESETS) + 1
    valid_choices = set(str(i) for i in range(1, max_choice + 1))
    
    while True:
        choice = Prompt.ask("Enter choice (3)", default="3")
        if choice in valid_choices:
            break
        console.print(f"[red]Invalid choice. Please enter 1-{max_choice}.[/red]")
    
    if int(choice) == len(REQUEST_COUNT_PRESETS) + 1:
        return IntPrompt.ask("Enter custom request count", default=100)
    
    return REQUEST_COUNT_PRESETS[int(choice) - 1]


def configure_text_modality() -> dict[str, Any]:
    """Configure text modality specific settings."""
    config = {"modality": "text"}
    
    console.print(Panel(
        "[cyan]Text Modality Configuration[/cyan]",
        border_style="cyan"
    ))
    
    config["input_tokens"] = select_token_length("Select Input Token Length:")
    config["output_tokens"] = select_token_length("Select Output Token Length (max_tokens):")
    
    return config


def configure_image_modality() -> dict[str, Any]:
    """Configure image modality specific settings."""
    config = {"modality": "image"}
    
    console.print(Panel(
        "[cyan]Image Modality Configuration[/cyan]\n"
        "[dim]Images will be randomly selected from the specified directory[/dim]",
        border_style="cyan"
    ))
    
    # Get image directory
    image_dir = Prompt.ask(
        "Enter path to image directory",
        default="./images"
    )
    config["image_directory"] = image_dir
    
    # Get max tokens for response
    config["max_tokens"] = select_token_length("Select Max Output Tokens:")
    
    return config


def configure_voice_modality() -> dict[str, Any]:
    """Configure voice modality specific settings."""
    config = {"modality": "voice"}
    
    console.print(Panel(
        "[cyan]Voice Modality Configuration[/cyan]\n"
        "[dim]Audio files will be randomly selected from the specified directory[/dim]",
        border_style="cyan"
    ))
    
    # Get audio directory
    audio_directory = Prompt.ask(
        "Enter path to sound directory",
        default="./sound"
    )
    config["audio_directory"] = audio_directory
    
    # Get max tokens for response
    config["max_tokens"] = select_token_length("Select Max Output Tokens:")
    
    return config


def prompt_export_results() -> tuple[bool, str | None]:
    """Prompt user if they want to export results."""
    console.print()
    
    should_export = Confirm.ask(
        "Would you like to export the results to a file? (y/n)",
        default=True
    )
    
    if not should_export:
        return False, None
    
    console.print("\n[bold]Select Export Format:[/bold]")
    console.print("  1. JSON (includes full raw metrics)")
    console.print("  2. CSV (raw request metrics only)")
    console.print()
    
    valid_choices = {"1", "2", "json", "csv"}
    while True:
        format_choice = Prompt.ask("Enter choice (1)", default="1").lower()
        if format_choice in valid_choices:
            break
        console.print("[red]Invalid choice. Please enter 1, 2, json, or csv.[/red]")
    
    if format_choice in ("1", "json"):
        return True, "json"
    else:
        return True, "csv"


def get_export_filename(format_type: str) -> str:
    """Prompt user for export filename."""
    from datetime import datetime
    
    default_name = f"llm_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
    default_path = Path("benchmarks") / default_name
    
    filename = Prompt.ask(
        "Enter filename for export",
        default=str(default_path)
    )

    filepath = Path(filename).expanduser()

    if not filepath.is_absolute() and filepath.parent == Path("."):
        filepath = Path("benchmarks") / filepath.name

    if filepath.suffix.lower() != f".{format_type}":
        filepath = filepath.with_suffix(f".{format_type}")

    return str(filepath)


def run_interactive_config() -> dict[str, Any]:
    """Run the full interactive configuration."""
    print_header()
    
    config = {}
    
    # 1. Select modality
    modality = select_modality()
    config["modality"] = modality
    
    # 2. API Configuration (with connection test)
    endpoint, api_route, available_models = select_api_config()
    config["endpoint"] = endpoint
    config["api_route"] = api_route
    
    # 3. Model selection (with available models if fetched)
    model = select_model(available_models)
    config["model"] = model
    
    # 4. Modality-specific configuration
    console.print()
    if modality == "text":
        modality_config = configure_text_modality()
    elif modality == "image":
        modality_config = configure_image_modality()
    elif modality == "voice":
        modality_config = configure_voice_modality()
    else:
        modality_config = {"modality": modality}
    
    config["modality_config"] = modality_config
    config["modality_config"]["model"] = model
    
    # 5. Load testing parameters
    console.print()
    console.print(Panel(
        "[cyan]Load Testing Parameters[/cyan]",
        border_style="cyan"
    ))
    
    config["concurrency"] = select_concurrency()
    config["target_rps"] = select_rps()
    config["total_requests"] = select_request_count()
    config["warmup_requests"] = 5  # Fixed as per requirements
    
    return config
