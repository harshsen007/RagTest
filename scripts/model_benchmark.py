
import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")

if not API_KEY:
    sys.exit(
        "ERROR: NVIDIA_API_KEY was not found.\n"
        "Add it to your .env file."
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
    timeout=300.0,
    max_retries=1,
)

# Exact model IDs shown in your NVIDIA catalog screenshot.
# If an ID fails, verify it against the API's model catalog.

MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "poolside/laguna-xs-2.1",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "minimaxai/minimax-m3.1-flash",
    "google/gemma-4-31b-it",
]


# Identical prompts and generation settings for every model.
SYSTEM_PROMPT = (
    "You are a precise coding assistant. "
    "Answer clearly and concisely."
)

USER_PROMPT = """
You are helping build a Python coding agent.

The agent can:
1. Read project files.
2. Write or modify code.
3. Run pytest.
4. Inspect test failures and fix the code.

Explain how you would implement a safe agent loop for this workflow.
Include a short Python example of a tool call to run pytest.
Keep your response under 250 words.
"""

TEMPERATURE = 0
MAX_TOKENS = 600
WIDTH = 84


# ============================================================
# 2. DISPLAY HELPERS
# ============================================================

def separator(char="-"):
    print(char * WIDTH)


def section(title):
    print()
    separator("=")
    print(f" {title}")
    separator("=")


def format_value(value, suffix=""):
    if value is None:
        return "N/A"
    return f"{value}{suffix}"


# ============================================================
# 3. BENCHMARK ONE MODEL
# ============================================================

def benchmark_model(model):
    section(f"MODEL: {model}")

    started = time.perf_counter()
    first_token_time = None
    response_text = []
    usage = None
    error = None

    print("\n[STREAMING RESPONSE]\n", flush=True)

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": USER_PROMPT,
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            # Some compatible APIs return usage in a final
            # chunk that contains no choices.
            if chunk.usage is not None:
                usage = chunk.usage

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Capture ordinary text output.
            content = delta.content

            if isinstance(content, str) and content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                response_text.append(content)
                print(content, end="", flush=True)

        print("\n")

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"\n\n[ERROR] {error}", flush=True)

    finally:
        # Close the stream if a request failed partway through.
        if "stream" in locals():
            try:
                stream.close()
            except Exception:
                pass

    finished = time.perf_counter()

    total_time = finished - started

    time_to_first_token = (
        first_token_time - started
        if first_token_time is not None
        else None
    )

    # A request that returned no text is not a successful
    # text-generation benchmark, even if no exception occurred.
    if error:
        status = "FAILED"
    elif not response_text:
        status = "EMPTY RESPONSE"
    else:
        status = "COMPLETED"

    # ========================================================
    # 4. PER-MODEL METRICS
    # ========================================================

    section(f"METRICS: {model}")

    print(f"{'Status':<30}: {status}")
    print(
        f"{'Time to first token':<30}: "
        f"{format_value(round(time_to_first_token, 2)
                        if time_to_first_token is not None
                        else None, ' s')}"
    )
    print(f"{'Total response time':<30}: {total_time:.2f} s")

    if usage is not None:
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        print(f"{'Input tokens (API)':<30}: {input_tokens}")
        print(f"{'Output tokens (API)':<30}: {output_tokens}")
        print(f"{'Total tokens (API)':<30}: {total_tokens}")
    else:
        input_tokens = None
        output_tokens = None
        total_tokens = None

        print(f"{'Input tokens (API)':<30}: N/A")
        print(f"{'Output tokens (API)':<30}: N/A")
        print(f"{'Total tokens (API)':<30}: N/A")
        print(
            "Note: The endpoint did not return token usage, "
            "or the stream failed before usage arrived."
        )

    if error:
        print(f"{'Error':<30}: {error}")

    return {
        "model": model,
        "status": status,
        "first_token_s": (
            round(time_to_first_token, 2)
            if time_to_first_token is not None
            else None
        ),
        "total_s": round(total_time, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "error": error,
    }


# ============================================================
# 5. FINAL COMPARISON TABLE
# ============================================================

def print_comparison(results):
    section("FINAL COMPARISON")

    columns = [
        ("Model", 34),
        ("Status", 15),
        ("TTFT(s)", 10),
        ("Total(s)", 11),
        ("Input", 9),
        ("Output", 9),
        ("Total", 9),
    ]

    header = "".join(
        f"{name:<{width}}"
        for name, width in columns
    )

    print(header)
    separator()

    for result in results:
        model_name = result["model"]

        if len(model_name) > 33:
            model_name = model_name[-33:]

        values = [
            model_name,
            result["status"],
            format_value(result["first_token_s"]),
            f"{result['total_s']:.2f}",
            format_value(result["input_tokens"]),
            format_value(result["output_tokens"]),
            format_value(result["total_tokens"]),
        ]

        print(
            f"{values[0]:<34}"
            f"{values[1]:<15}"
            f"{values[2]:>10}"
            f"{values[3]:>11}"
            f"{values[4]:>9}"
            f"{values[5]:>9}"
            f"{values[6]:>9}"
        )

    separator()
    completed = sum(
        result["status"] == "COMPLETED"
        for result in results
    )

    print(f"Successful text responses: {completed}/{len(results)}")
    print("Token counts are API-reported, not estimated.")


# ============================================================
# 6. RUN MODELS SEQUENTIALLY
# ============================================================

def main():
    results = []

    section("NVIDIA MODEL BENCHMARK")

    print(f"Models to test : {len(MODELS)}")
    print(f"Temperature    : {TEMPERATURE}")
    print(f"Max output     : {MAX_TOKENS} tokens")
    print("Execution      : Sequential")
    print("Streaming      : Enabled")
    print("Usage reporting: Requested")

    for index, model in enumerate(MODELS, start=1):
        print(f"\nRunning model {index}/{len(MODELS)}")
        result = benchmark_model(model)
        results.append(result)

    print_comparison(results)

    section("BENCHMARK FINISHED")
    print(
        "Compare response quality, tool-call reliability, "
        "latency, and token usage before choosing a model."
    )


if __name__ == "__main__":
    main()
