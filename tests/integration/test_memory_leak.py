# SPDX-License-Identifier: Apache-2.0
"""
Integration test for memory leak after cache saving.
Run with:
    pytest tests/integration/test_memory_leak.py -v -m slow -s
"""

import gc
import time
from pathlib import Path

import psutil
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]
SERVER_URL = "http://localhost:51234"
MODEL_NAME = "gemma-4-26B-A4B-it-oQ3-fp16"
MEMORY_INCREASE_THRESHOLD_GB = 2.0
CACHE_SAVE_WAIT_SECONDS = 5


def get_omlx_server_memory_mb() -> float:
    """Find omlx server process and get its RAM usage in MB."""

    # Find processes that could be the omlx server
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline)

            # Match common omlx server patterns
            if (
                any(
                    pattern in cmdline_str
                    for pattern in [
                        "omlx",
                    ]
                )
                and "pytest" not in cmdline_str
            ):
                # Found the server process
                mem_bytes = proc.memory_info().rss
                return mem_bytes / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    raise RuntimeError("Could not find omlx server process")


def send_request(prompt: str, max_tokens: int = 20) -> dict:
    """Send chat completion request to server."""
    import httpx

    response = httpx.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
        timeout=300.0,
    )
    response.raise_for_status()
    return response.json()


def test_ram_stable_after_decode():
    """
    Test that RAM usage stays stable after prefill -> decode.

    Acceptance: Up to 2GB increase is acceptable after cache saving.
    """
    prompt_file = Path("/tmp/long_prompt.txt")
    assert prompt_file.exists(), "Create /tmp/long_prompt.txt with ~260k chars"

    prompt = prompt_file.read_text()
    print(f"\n[TEST] Prompt length: {len(prompt)} chars")

    mem_before = get_omlx_server_memory_mb()
    print(f"[TEST] Memory before: {mem_before:.1f} MB")

    # Run prefill + decode
    response = send_request(prompt, max_tokens=20)
    print(f"[TEST] Response: {response['choices'][0]['message']['content'][:50]}...")

    # Wait for cache saving to complete
    time.sleep(CACHE_SAVE_WAIT_SECONDS)
    gc.collect()

    mem_after = get_omlx_server_memory_mb()
    print(f"[TEST] Memory after: {mem_after:.1f} MB")

    increase_gb = (mem_after - mem_before) / 1024
    print(f"[TEST] Memory increase: {increase_gb:.2f} GB")

    assert increase_gb <= MEMORY_INCREASE_THRESHOLD_GB, (
        f"Memory increased by {increase_gb:.2f}GB (expected <= {MEMORY_INCREASE_THRESHOLD_GB}GB)"
    )
