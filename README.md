# Multi-Modal-LLM-Benchmark

An asynchronous CLI benchmarker for OpenAI-compatible LLM APIs across text, image, and voice workflows.

Python 3.10+ is required. The examples below use Python 3.12.

## Features

- **Multiple Modalities**: Supports Text, Image (multimodal), and Voice inputs
- **Streaming Metrics**: Accurately measures TTFT, TPOT, and throughput via streaming
- **Configurable Load**: Adjustable concurrency, RPS targeting, and request counts
- **Warm-up Phase**: Pre-test requests to eliminate cold-start latency
- **Rich Output**: Beautiful terminal tables with detailed metrics
- **Data Export**: Export results to JSON or CSV for further analysis

## Installation

```bash
# Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Interactive Mode

Run the tool and follow the prompts:

```bash
# Run from project root
python -m llm_load_tester

# Or use the convenience script
./run.sh
```

### Configuration Options

The interactive CLI will guide you through:

1. **Modality Selection**: Text, Image, or Voice
2. **API Configuration**: Endpoint, route, and model name (with presets)
3. **Modality-specific Settings**:
   - Text: Input/output token lengths
   - Image: Directory path containing images
   - Voice: Directory path containing audio files
4. **Load Parameters**: Concurrency, target RPS, total requests

## Metrics Collected

- **TTFT** (Time To First Token): Latency until first response chunk
- **TPOT** (Time Per Output Token): Average generation speed per token
- **End-to-End Latency**: Total request completion time (p50, p95, p99)
- **Throughput**: Tokens per second (overall and per-request averages)
- **Error Rates**: Categorized by HTTP status codes

## Project Structure

```
.
├── benchmarks/       # Export destination for benchmark results
├── images/           # Local image inputs (gitignored except .gitkeep)
├── llm_load_tester/  # Python package
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── run.sh
└── sound/            # Local audio inputs (gitignored except .gitkeep)
```

## Example Workflow

```bash
# 1. Start the tool
python -m llm_load_tester

# 2. Select modality (e.g., Text)
# 3. Choose API preset or enter custom endpoint
# 4. Select or enter model name
# 5. Set token lengths (e.g., 512 input, 1024 output)
# 6. Configure load (e.g., 8 concurrent, 5 RPS, 100 requests)
# 7. Wait for warm-up and benchmark completion
# 8. View results table in terminal
# 9. Optionally export to JSON/CSV
```

## API Compatibility

This tool uses the OpenAI-compatible chat completions API format:
- Streaming via `stream: true`
- SSE (Server-Sent Events) response parsing
- Standard message format for text
- Base64-encoded media for image/voice

Compatible with:
- vLLM
- Text Generation Inference (TGI)
- TensorRT-LLM
- LMStudio
- Ollama
- Any OpenAI-compatible endpoint

## Notes

- The tool generates dummy text for text modality testing
- For image testing, place files in `images/` or another directory and the runner will cycle through a shuffled pool
- For voice testing, place files in `sound/` or another directory and the runner will cycle through a shuffled pool
- Warm-up requests (5) are sent before timing to initialize model/VRAM
- Local assets and generated benchmark results are intentionally ignored by git
