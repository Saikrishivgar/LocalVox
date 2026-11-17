"""Independent CLI verification script for Ollama local LLM (Component C)."""

import argparse
import json
import sys

from backend.config import get_config
from backend.llm.ollama_client import OllamaClient, OllamaError

DEFAULT_SUITE: list[tuple[str, list[dict[str, str]]]] = [
    (
        "en_factual",
        [
            {
                "role": "user",
                "content": "What is the powerhouse of a eukaryotic cell? Answer in 1 sentence.",
            }
        ],
    ),
    (
        "en_educational",
        [
            {
                "role": "user",
                "content": (
                    "Explain photosynthesis clearly for a high school student in 3 short sentences."
                ),
            }
        ],
    ),
    (
        "ta_conversation",
        [
            {
                "role": "user",
                "content": "வணக்கம்! நியூட்டனின் முதல் விதியை எளிய தமிழில் 2 வாக்கியங்களில் விளக்கவும்.",
            }
        ],
    ),
    (
        "en_conversation",
        [
            {
                "role": "user",
                "content": "Hi, I have a biology exam tomorrow. How should I revise tonight?",
            }
        ],
    ),
    (
        "ta_en_mixed",
        [
            {
                "role": "user",
                "content": "Binary search algorithm எப்படி work ஆகுது nu simple-ஆ explain பண்ணுங்க.",
            }
        ],
    ),
    (
        "multi_turn_context",
        [
            {
                "role": "user",
                "content": "I am studying human blood circulation. Keep track of my topic.",
            },
            {
                "role": "assistant",
                "content": (
                    "Got it! We are focusing on human blood circulation. "
                    "Ask away whenever you are ready."
                ),
            },
            {"role": "user", "content": "What does the left ventricle pump blood to?"},
            {
                "role": "assistant",
                "content": (
                    "The left ventricle pumps oxygenated blood through the aorta "
                    "to the rest of the body."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize what topic we started with and what the left ventricle does "
                    "in one sentence."
                ),
            },
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Ollama LLM verification."""
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Test local Ollama LLM inference.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to execute.")
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run full Phase 1 multilingual prompt suite.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=cfg.llm.model_name,
        help="Local Ollama model name.",
    )
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=cfg.llm.think,
        help="Enable or disable Qwen3 thinking mode.",
    )
    parser.add_argument("--max-tokens", type=int, default=cfg.llm.max_tokens)
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser.parse_args()


def main() -> int:
    """Check Ollama, verify model availability, run prompts, and report metrics."""
    args = parse_args()
    cfg = get_config()
    client = OllamaClient(cfg)

    try:
        model_info = client.verify_model_exists(args.model)
    except OllamaError as exc:
        print(f"LLM setup error: {exc}", file=sys.stderr)
        return 1

    size_bytes = model_info.get("size")
    size_mb = round(float(size_bytes) / (1024.0 * 1024.0), 2) if size_bytes else None

    if args.suite:
        runs = DEFAULT_SUITE
    elif args.prompt:
        runs = [("custom", [{"role": "user", "content": args.prompt}])]
    else:
        runs = [DEFAULT_SUITE[0]]

    results_out = []
    for name, turns in runs:
        messages = [{"role": "system", "content": cfg.llm.system_prompt}, *turns]
        try:
            res = client.chat(
                messages,
                model_name=args.model,
                think=args.think,
                max_tokens=args.max_tokens,
            )
        except OllamaError as exc:
            print(f"LLM inference error on '{name}': {exc}", file=sys.stderr)
            return 1
        running_mem_mb = client.get_running_memory_mb(args.model)
        results_out.append(
            {
                "case": name,
                "model_size_mb": size_mb,
                "ollama_memory_mb": running_mem_mb,
                **res.model_dump(),
            }
        )

    if args.json:
        print(json.dumps(results_out, indent=2, ensure_ascii=False))
        return 0

    for item in results_out:
        print(f"=== Case: {item['case']} ({item['model_name']}, think={args.think}) ===")
        print(f"Model Disk Size:  {item['model_size_mb']} MiB")
        print(f"Ollama Memory:    {item['ollama_memory_mb']} MiB")
        print(f"TTFT:             {item['ttft_ms']} ms")
        print(f"Total Time:       {item['total_time_ms']} ms")
        print(f"Load Duration:    {item['load_duration_ms']} ms")
        print(f"Tokens (in/out):  {item['prompt_tokens']} / {item['completion_tokens']}")
        print(f"Throughput:       {item['tokens_per_sec']} tok/s")
        print(f"Response:\n{item['response_text']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
