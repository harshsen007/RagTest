
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

print("Checking NVIDIA API model catalog...\n")

try:
    models = client.models.list()

    keywords = [
        "nemotron-3-ultra-550b-a55b",
        "laguna-xs-2.1",
        "nemotron-3.5-super-vl-120b-a12b",
        "minimax-m3.1-flash",
        "gemma-4-31b-it",
    ]

    for keyword in keywords:
        matches = [
            model.id
            for model in models.data
            if keyword in model.id.lower()
        ]

        print(f"\n{keyword}")

        if matches:
            for model_id in matches:
                print(f"  API ID: {model_id}")
        else:
            print("  No matching ID found in this catalog.")

except Exception as exc:
    print(f"Catalog request failed: {type(exc).__name__}: {exc}")
