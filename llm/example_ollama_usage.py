#!/usr/bin/env python3
"""Example: using OllamaClient for text and vision tasks.

Requires a running Ollama instance with at least one model pulled.

    ollama pull llama3
    ollama pull llava          # optional — for image analysis

Run::

    python -m llm.example_ollama_usage
"""

from __future__ import annotations

import sys
from pathlib import Path

from llm.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    VisionModelUnavailableError,
)
from utils.logger import setup_logging


def main() -> None:
    setup_logging()

    # ── Create client from config.yaml ───────────────────────────────────
    client = OllamaClient.from_config()
    print(f"Client: {client}\n")

    # ── 1. Check connection ──────────────────────────────────────────────
    print("1. Checking Ollama connection...")
    if not client.check_connection():
        print("   ERROR: Ollama is not reachable. Start it with: ollama serve")
        sys.exit(1)
    print("   OK — Ollama is running.\n")

    # ── 2. Text generation ───────────────────────────────────────────────
    print("2. Text generation (generate):")
    reply = client.generate(
        "In 2 sentences, explain what a Fourier transform is.",
        system="You are a concise university tutor.",
    )
    print(f"   {reply}\n")

    # ── 3. Chat ──────────────────────────────────────────────────────────
    print("3. Multi-turn chat (chat):")
    messages = [
        {"role": "system", "content": "You are a helpful tutor."},
        {"role": "user", "content": "What is the difference between FFT and DFT?"},
    ]
    reply = client.chat(messages)
    print(f"   {reply}\n")

    # ── 4. Vision (optional) ────────────────────────────────────────────
    image_path = Path("output/assets/images/sample.png")
    if image_path.is_file():
        print("4. Vision analysis (generate_with_image):")
        try:
            description = client.generate_with_image(
                "Describe this image and explain its academic relevance.",
                image_path,
            )
            print(f"   {description}\n")
        except VisionModelUnavailableError as exc:
            print(f"   Vision model unavailable: {exc}")
            print("   Skipping image analysis (fallback).\n")
    else:
        print("4. Vision: no sample image found — skipping.\n")

    print("Done.")


if __name__ == "__main__":
    main()
