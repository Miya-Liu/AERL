#!/usr/bin/env python3
"""
Runner script for image_qa_code.json tasks.
Processes image-based QA tasks using backend_run.py
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

from customized_areal.tpfc.backend_run import run_backend, DEFAULT_BACKEND_AUTH_TOKEN, DEFAULT_REFRESH_TOKEN


async def process_single_task(task_data, base_image_path, user_id=None, model_name=None, base_url=None, api_key=None, backend_auth_token=None, refresh_token=None):
    """Process a single image QA task."""

    question = task_data.get("question", "")
    golden_answer = task_data.get("golden_answer", "")
    file_path = task_data.get("file_path", "")
    max_hops = task_data.get("max_hops", 3)
    trajectory = task_data.get("trajectory", [])

    # Construct full image path
    if file_path:
        full_image_path = os.path.join(base_image_path, os.path.basename(file_path))
    else:
        full_image_path = ""

    # Prepare task description with question and image info
    task_description = f"""Question: {question}

Image path: {full_image_path if full_image_path else 'No image provided'}

Max hops: {max_hops}

Please answer this question based on the image provided."""

    try:
        messages = await run_backend(
            task_description=task_description,
            task_file_path=[full_image_path] if full_image_path else [],
            tags=["image_qa", f"max_hops_{max_hops}"],
            user_id=user_id or "13183c90-ac94-403e-893e-c53552ad429d",
            model_name=model_name or "openrouter/qwen/qwen3-235b-a22b",
            base_url=base_url,
            api_key=api_key,
            backend_auth_token=backend_auth_token,
            refresh_token=refresh_token,
            gt=golden_answer,
        )

        return {
            "question": question,
            "golden_answer": golden_answer,
            "predicted_answer": messages,
            "file_path": file_path,
            "status": "success"
        }
    except Exception as e:
        return {
            "question": question,
            "golden_answer": golden_answer,
            "predicted_answer": None,
            "file_path": file_path,
            "status": "error",
            "error": str(e)
        }


async def main(
    json_path="/dfs/share-groups/foundationmodelgroup/LRM/multimodaldata/image/image_qa_code.json",
    base_image_path="/dfs/share-groups/foundationmodelgroup/LRM/multimodaldata/image/image_data",
    output_path="./image_qa_results.json",
    limit=None,
    user_id=None,
    model_name=None,
    base_url=None,
    api_key=None,
    backend_auth_token=None,
    refresh_token=None,
):
    """Main function to process image QA tasks."""

    # Load tasks from JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} tasks from {json_path}")

    # Limit tasks if specified
    if limit:
        tasks = tasks[:limit]
        print(f"Processing first {limit} tasks")

    results = []

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Processing task...")
        print(f"    Question: {task.get('question', '')[:100]}...")

        result = await process_single_task(
            task_data=task,
            base_image_path=base_image_path,
            user_id=user_id,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            backend_auth_token=backend_auth_token,
            refresh_token=refresh_token,
        )

        results.append(result)

        # Save intermediate results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"    Status: {result['status']}")
        if result['status'] == 'success':
            print(f"    Answer: {result['predicted_answer']}")
        else:
            print(f"    Error: {result.get('error', 'Unknown error')}")

    print(f"\n\nCompleted! Results saved to {output_path}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'error')}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run image QA tasks")
    parser.add_argument("--json-path", default="/dfs/share-groups/foundationmodelgroup/LRM/multimodaldata/image/image_qa_code.json",
                        help="Path to image_qa_code.json")
    parser.add_argument("--base-image-path", default="/dfs/share-groups/foundationmodelgroup/LRM/multimodaldata/image/image_data",
                        help="Base directory containing images")
    parser.add_argument("--output", default="./image_qa_results.json",
                        help="Output path for results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tasks to process")
    parser.add_argument("--user-id", default=None,
                        help="User ID for database")
    parser.add_argument("--model-name", default="Qwen3.5_397B_A17B_FP8",
                        help="Model name to use")
    parser.add_argument("--base-url", default="http://10.254.10.192:8443/service-large-64-1772072563849/llm/v1",
                        help="Base URL for proxy")
    parser.add_argument("--api-key", default="RlkgHzgBa2zbPcQrw96rdn68dr7kXk8Mv84Nhs5Trrk6gfq8Cqw5CcDQ787p6d6bPrAWDrw8b97gq8N7jWWxhasnVP76FD76r8tJ2688mdPnSP7N6V7gl1VLKD9LasJq",
                        help="API key for proxy")
    parser.add_argument("--backend-auth-token", default=DEFAULT_BACKEND_AUTH_TOKEN,
                        help="Real Supabase JWT access token for backend API authentication (required for le-agent-dev2)")
    parser.add_argument("--refresh-token", default=DEFAULT_REFRESH_TOKEN,
                        help="Supabase refresh token to automatically renew access token when expired")

    # parser.add_argument("--base-url", default="https://modelfactory.lenovo.com/service-large-544-1773728352034/llm/v1",
    #                     help="Base URL for proxy")
    # parser.add_argument("--api-key", default="Rl44TWGlj7Nn06txRhLrmgLf888A768jvxZc6Xm1gD7mtcrz2Vrg0pNH8rdP8mg688jl8Xdcq7MSB7Anzp8pf8XgnK7168R2267ZBS5dSlzbGhr6rwB5t6ZcP5wn6w7t",
    #                     help="API key for proxy")
    args = parser.parse_args()

    asyncio.run(main(
        json_path=args.json_path,
        base_image_path=args.base_image_path,
        output_path=args.output,
        limit=args.limit,
        user_id=args.user_id,
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        backend_auth_token=args.backend_auth_token,
        refresh_token=args.refresh_token,
    ))
