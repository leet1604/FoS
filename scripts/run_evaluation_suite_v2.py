from __future__ import annotations

import argparse
import json

from evaluation.runners.policy_adapters import PolicyConfigV2
from evaluation.runners.suite_v2 import (
    load_action_spaces,
    load_episodes_v2,
    run_suite_frozen,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FoS policies over frozen v2 action spaces.")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--action-spaces", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--policies",
        default="seed_only,greedy,tool_only,full_agent",
        help="Comma-separated policy names.",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--min-evidence-support", type=int, default=2)
    parser.add_argument("--candidate-top-k", type=int, default=5)
    parser.add_argument("--disable-provisional", action="store_true")
    parser.add_argument("--full-agent-backend", choices=["heuristic", "chat"], default="heuristic")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-sec", type=int, default=30)
    args = parser.parse_args()

    config = PolicyConfigV2(
        min_evidence_support=args.min_evidence_support,
        allow_provisional=not args.disable_provisional,
        candidate_top_k=args.candidate_top_k,
        full_agent_backend=args.full_agent_backend,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_sec=args.timeout_sec,
    )
    results = run_suite_frozen(
        load_episodes_v2(args.episodes),
        load_action_spaces(args.action_spaces),
        policy_names=[x.strip() for x in args.policies.split(",") if x.strip()],
        output_dir=args.output_dir,
        policy_config=config,
        max_episodes=args.max_episodes,
    )
    print(json.dumps({"n_runs": len(results), "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
