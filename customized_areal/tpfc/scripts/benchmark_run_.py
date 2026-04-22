import asyncio
import os

import dotenv
from omegaconf import OmegaConf

from customized_areal.tpfc.scripts.benchmark_run import entrypoint

def main():
    dotenv.load_dotenv()

    # Create configuration using OmegaConf
    cfg = OmegaConf.create(
        {
            "benchmark": {
                "name": "gaia-validation",
                "data": {
                    "data_dir": "customized_areal/dataset/gaia-benchmark/gaia/2023/validation",
                    "metadata_file": "metadata.jsonl",
                    "whitelist": [],
                },
                "execution": {"max_concurrent": 10, "max_tasks": 166, "pass_at_k": 1},
            },
            "llm": {
                "provider": "openai",
                # "model_name": "openrouter/gpt-5",
                # "model_name": "openai-compatible/gpt-5",
                # "model_name": "openrouter/qwen/qwen3.5-9b",
                "model_name": "openrouter/qwen/qwen3-vl-8b-thinking",
                # "model_name": "openrouter/qwen/qwen3-32b",
                "enable_thinking": False,
                "reasoning_effort": "low",
                "stream": False,
            },
            "env": {"openai_api_key": ""},
            "level": 1,
            "user_id": "62ec5137-d121-4c8c-b175-ee165bdf38e4",
            "agent_id": os.environ.get("main_agent_id", ""),
            "backend_mode": True,
            "base_url": "http://10.254.245.58:8443/service-large-544-1763113682810/llm/v1",  # Set your proxy base URL here or via CLI
            "api_key": "xp77r4bxFv81bPcrA7kK6j77HtDrmFcl7Knm7M68FPrkzFWnzAclZ2jqR2kThPaarChv786dkpTS7Za0XpXw7wL7bDl77181LQrw7g5j7kC8MxkZ6RGnkwG728TS778V",  # Set your API key here or via CLI
            # "base_url": "http://10.254.245.58:8443/service-large-544-1763113682810/llm/v1",  # Set your proxy base URL here or via CLI
            # "api_key": "xp77r4bxFv81bPcrA7kK6j77HtDrmFcl7Knm7M68FPrkzFWnzAclZ2jqR2kThPaarChv786dkpTS7Za0XpXw7wL7bDl77181LQrw7g5j7kC8MxkZ6RGnkwG728TS778V",   # Set your API key here or via CLI
        }
    )

    # Compute derived values
    cfg.tags = [
        f"{cfg.benchmark.name}",
        f"{cfg.llm.model_name}",
        "trained_retry_think_fixtool_0421",
        # "compression_1w",
        f"level_{cfg.level}",
    ]
    cfg.output_dir = f"logs/{'-'.join(cfg.tags[:-1])}/{cfg.tags[-1]}"

    asyncio.run(entrypoint(cfg))


if __name__ == "__main__":
    main()
