# Evaluation datasets

- `*.template.jsonl` files are schemas/examples only and are not scientific results.
- Use `scripts/build_eval_set_from_pair_cache.py` to create a development positive set.
- Keep `*_hidden_oracle.jsonl` outside the agent's runtime input path.
- Manually curate negative, low-evidence, and safety episodes before reporting a full task-success rate.
