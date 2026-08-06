# FoS — Findings of Selectivity

**Evidence-constrained agentic AI for target-selective molecular optimization**

FoS is a molecular optimization agent that searches for structural changes that preserve or improve **on-target activity** while reducing **off-target activity**.  
The LLM is used as a **decision policy**, not as a free-form molecule generator or an activity predictor.

> Current version: **v0.7.5 — proposal-ready pilot**  
> Status: Stage A/B/C integration, retrospective measured evaluation, controlled behavior benchmark, and reproducible reporting are implemented.

---

## Overview

Lead optimization is not a one-shot generation problem. It is an iterative process that requires the system to:

1. collect target-pair-specific evidence,
2. identify chemically applicable transformations,
3. choose the next action under uncertainty,
4. reject unsafe or weakly supported candidates,
5. recover from failed paths, and
6. retain an auditable optimization trajectory.

FoS addresses this as an **evidence-constrained planning problem**.

```text
Seed molecule + on-target + optional off-target
                    │
                    ▼
       Stage A — Evidence construction
       ChEMBL activity · analogs · MMP rules
                    │
                    ▼
       Stage B — Agentic optimization loop
       Observe → Plan → Execute → Evaluate → Update
                    │
                    ▼
       Stage C — Independent validation and reranking
       Chemistry · evidence · optional QSAR/docking
                    │
                    ▼
       Final candidate + decision trace + audit report
```

---

## Core design principles

### 1. The LLM does not freely generate molecules

The Stage B policy selects among candidates and transformations that have already passed chemical applicability checks. Candidate construction and structural validation are handled by RDKit and rule-based tools.

### 2. Evidence defines the action space

FoS primarily uses:

- measured activity data from ChEMBL,
- matched molecular pair (MMP) transformations,
- transformation support counts,
- observed activity changes,
- direction consistency,
- dispersion and provenance,
- candidate-centered local evidence graphs.

An unsupported edit is not treated as a valid optimization action.

### 3. Missing evidence is not interpreted as zero effect

Unknown off-target activity remains unknown. FoS can route the candidate to additional evidence collection, place it in a validation queue, abstain, or terminate the search.

### 4. Optimization and final validation are separated

The Stage B Neighbor-KNN surrogate is an exploration aid and is not considered independent proof. Stage C can consume independently generated QSAR or docking results, but docking is treated as corroborative evidence rather than a replacement for potency/selectivity measurements.

### 5. Every decision is auditable

FoS records:

- selected transformation,
- parent and child molecules,
- evidence used,
- predicted or measured activity changes,
- accepted, rejected, or deferred decisions,
- Critic/Reflection outputs,
- tool calls and failures,
- termination reasons,
- run configuration and timing.

---

## Optimization objective

For molecule \(x\), worst-case selectivity is defined as:

\[
S(x) = pAct_{\mathrm{on}}(x) -
\max_j pAct_{\mathrm{off},j}(x)
\]

The improvement from the seed molecule is:

\[
\Delta S = S(x_{\mathrm{candidate}}) - S(x_{\mathrm{seed}})
\]

A candidate is not considered successful solely because \(\Delta S\) is positive. FoS also checks:

- on-target retention,
- chemical validity,
- required safety constraints,
- evidence sufficiency,
- procedural integrity.

---

## Architecture

## Stage A — Evidence construction

Stage A prepares the target-pair-specific environment in which the agent is allowed to act.

### Main responsibilities

- SMILES/SELFIES normalization and canonicalization
- ligand-centric off-target discovery
- optional user-provided off-target hints
- ChEMBL activity retrieval
- repeated measurement aggregation by median
- uncertainty estimation from activity dispersion
- analog and MMP extraction
- transformation-level support and sign-consistency aggregation
- target and target-pair evidence caching
- candidate-centered dynamic local graph construction
- provenance tracking outside the graph

### Off-target modes

| Mode | Description |
|---|---|
| `hint_only` | Use only the user-provided off-target |
| `hint_plus_auto` | Preserve provided hints and add automatically discovered targets |
| `auto` | Discover off-target candidates from the molecule and on-target |

### Evidence storage principle

Raw activity records and supporting molecular pairs are stored outside the graph. The graph contains only decision-relevant aggregated relations, while provenance IDs connect each summary back to the underlying evidence.

```text
Molecule ── aggregated_activity ──> Target

Candidate ── applicable_rule ──> MMP rule
MMP rule ── generates ──> Product
```

---

## Stage B — Evidence-gated optimization loop

Stage B runs an iterative molecular optimization loop.

```text
Observe current state
        ↓
Plan transformation and tool use
        ↓
Generate and validate candidates
        ↓
Evaluate selectivity, retention, evidence, and constraints
        ↓
Accept / Reject / Defer / Expand evidence
        ↓
Update state or terminate
```

### Supported search modes

| Mode | Behavior |
|---|---|
| `beam` | Maintains multiple candidate branches |
| `trajectory` | Follows a contiguous optimization path for easier audit and visualization |

### Candidate states

- `accepted_candidates`
- `validation_queue`
- `rejected_candidates`
- `active_path`

### Recovery behavior

When a proposal fails, the agent can:

- retry with another transformation,
- expand evidence,
- switch branches,
- backtrack,
- abstain,
- terminate with `budget_exhausted` or another explicit reason.

### Provisional trajectory

v0.7.5 supports bounded provisional movement through a promising MMP candidate when evidence is useful for exploration but insufficient for final validation.

A provisional Stage B state is not automatically promoted to a validated Stage C result.

---

## Stage C — Conservative validation and reranking

Stage C receives the Stage B beam or trajectory tip and applies final chemistry/evidence checks.

### Decision labels

- `SUPPORTED`
- `SUPPORTED_COMPUTATIONAL`
- `NEEDS_VALIDATION`
- `REJECTED`

### Optional external inputs

- independent QSAR or pretrained predictor output
- docking summary
- externally generated validation JSON

Without independent evidence, a promising Stage B candidate can remain `NEEDS_VALIDATION`.

---

## v0.7.5 evaluation framework

FoS v0.7.5 evaluates both molecular outcomes and agent behavior.

### Track A — Retrospective measured benchmark

A held-out measured endpoint is kept outside the agent loop.

```text
Visible evidence only
        ↓
Visible-only MMP construction
        ↓
Frozen candidate action space
        ↓
Agent or baseline selection
        ↓
Private measured oracle lookup
        ↓
Episode metrics
```

Implemented split strategies include:

- document-time split,
- leave-one-document-out split,
- scaffold split.

Leakage auditing and checksum-based release freezing are included.

### Track B — Controlled behavior benchmark

Controlled fixtures test whether the agent behaves correctly when molecular optimization is impossible, unsafe, uncertain, or interrupted.

Included behavior types:

- positive decision,
- no valid move,
- low evidence,
- safety challenge,
- tool failure,
- budget exhaustion.

These fixtures evaluate agent control behavior and are not biological activity benchmarks.

### Track C — Generalization schema

The evaluation schema supports target and scaffold holdout labels for future multi-pair generalization experiments.

---

## Evaluation policies

The frozen-action-space runner supports:

| Policy | Description |
|---|---|
| `seed_only` | Return the seed without optimization |
| `random_valid` | Randomly select a valid action |
| `greedy` | Select the highest predicted selectivity candidate |
| `tool_only` | Use a deterministic evidence/tool heuristic |
| `full_agent` | Use the FoS policy through heuristic or chat backend |

The heuristic `full_agent` mode is intended for pipeline and evaluation smoke tests. It does not demonstrate superiority of an LLM policy.

---

## Evaluation metrics

### Molecular performance

- Qualified Task Success Rate
- Final worst-case \(\Delta S\)
- On-target retention \(\Delta on\)
- Oracle regret
- Oracle coverage
- Positive Step Rate

### Safe and uncertain decision-making

- Unsafe Acceptance Rate
- Correct Rejection Rate
- Correct Abstention Rate

### Agent procedure

- procedural integrity
- recovery rate
- trajectory continuity
- cycle detection
- termination correctness

### Resource efficiency

- total tool calls
- wall-clock time
- \(\Delta S\) per call

---

## Current pilot status

The v0.7.5 repository includes a proposal-ready pilot for the EGFR–HER2 pair.

| Item | Current pilot |
|---|---:|
| On-target | EGFR (`CHEMBL203`) |
| Off-target | HER2 (`CHEMBL1824`) |
| Paired compounds | 1,547 |
| ChEMBL documents | 619 |
| Bemis–Murcko scaffolds | 675 |
| Measured split | Leave-one-document-out |
| Real measured episodes | 1 |
| Controlled behavior fixtures | 12 |
| Repeated policy runs | 156 |
| Leakage audit | Pass |

A prototype trajectory run updated the molecule four times and changed the internal selectivity score from \(S=-3.30\) to \(S=4.02\), corresponding to \(\Delta S=+7.32\). This result is presented as an end-to-end pilot demonstrating that the optimization loop runs and records a continuous trajectory; it is not a claim of general benchmark performance.

> v0.7.5 is a **proposal-ready pilot**, not a completed scientific benchmark.  
> The measured replay currently contains one real held-out episode.

---

## Installation

### Requirements

- Python 3.11 or later
- RDKit
- access to ChEMBL for live evidence retrieval
- optional: Ollama or another OpenAI-compatible chat endpoint

### Editable installation

```bash
git clone https://github.com/leet1604/FoS.git
cd FoS

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

Run the test suite:

```bash
PYTHONPATH=src pytest -q
```

---

## Quick start

## 1. Offline fixture

Use the fixture workflow to verify the Stage B loop without live ChEMBL retrieval.

```bash
PYTHONPATH=src python scripts/run_stage_b_fixture_v2.py
```

---

## 2. Known target pair: Stage A → B → C

This is the recommended bounded end-to-end workflow when the on-target and off-target are already known.

```bash
PYTHONPATH=src python scripts/run_stage_abc_live_pair.py \
  --seed-smiles "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1" \
  --on-target CHEMBL203 \
  --off-target CHEMBL1824 \
  --model qwen3:8b \
  --base-url http://127.0.0.1:11434/v1 \
  --neighbor-surrogate \
  --dynamic-discovery \
  --search-mode trajectory \
  --max-provisional-depth 2 \
  --max-iterations 5 \
  --output-prefix outputs/stage_abc/egfr_her2
```

Outputs:

```text
outputs/stage_abc/
├── egfr_her2_stage_b.json
├── egfr_her2_stage_c.json
└── egfr_her2_stage_c.md
```

Run without a chat model:

```bash
PYTHONPATH=src python scripts/run_stage_abc_live_pair.py \
  --seed-smiles "<SMILES>" \
  --on-target CHEMBL203 \
  --off-target CHEMBL1824 \
  --heuristic \
  --neighbor-surrogate \
  --search-mode trajectory \
  --output-prefix outputs/stage_abc/heuristic_run
```

---

## 3. Warm a target-pair cache

Precompute the evidence cache for a known pair before repeated experiments.

```bash
PYTHONPATH=src python scripts/warm_pair_cache.py \
  --seed-smiles "<SMILES>" \
  --on-target CHEMBL203 \
  --off-target CHEMBL1824 \
  --output outputs/cache_warm_manifest.json
```

---

## 4. Automatic off-target discovery: fast profile

The fast profile restricts evidence collection and optimization depth for quick feedback.

Current bounded settings include:

- maximum 10 analogs,
- one selected off-target,
- three Stage B iterations,
- final top-1 candidate.

```bash
PYTHONPATH=src python scripts/run_stage_b_live_fast.py \
  --seed-smiles "<SMILES>" \
  --on-target CHEMBL203 \
  --model qwen3:8b \
  --base-url http://127.0.0.1:11434/v1 \
  --neighbor-surrogate \
  --dynamic-discovery \
  --search-mode trajectory \
  --output outputs/stage_b_live_fast/result.json
```

---

## 5. Automatic off-target discovery: full profile

The full profile performs broader evidence collection and should be reserved for final runs.

Current settings include:

- maximum 60 analogs,
- up to three selected off-targets,
- six Stage B iterations,
- final top-3 candidates.

```bash
PYTHONPATH=src python scripts/run_stage_b_live_full.py \
  --seed-smiles "<SMILES>" \
  --on-target CHEMBL203 \
  --model qwen3:8b \
  --base-url http://127.0.0.1:11434/v1 \
  --neighbor-surrogate \
  --dynamic-discovery \
  --search-mode trajectory \
  --output outputs/stage_b_live_full/result.json
```

---

## 6. Add independent Stage C predictions

Run Stage C on an existing Stage B result:

```bash
PYTHONPATH=src python scripts/run_stage_c.py \
  --stage-b-result outputs/stage_b_live_pair/result.json \
  --output outputs/stage_c/result.json
```

Add independent prediction and docking exports:

```bash
PYTHONPATH=src python scripts/run_stage_c.py \
  --stage-b-result outputs/stage_b_live_pair/result.json \
  --prediction-json outputs/external/independent_qsar.json \
  --docking-json outputs/external/docking_summary.json \
  --output outputs/stage_c/validated_result.json
```

---

## Reproduce the v0.7.5 proposal evaluation

The included notebook provides the easiest guided workflow:

```text
FoS_v075_proposal_evaluation_colab.ipynb
```

The command-line workflow is summarized below.

### 1. Build controlled behavior fixtures

```bash
PYTHONPATH=src python scripts/build_behavior_benchmark.py \
  --output-dir examples/evaluation_v075/behavior_release \
  --repeats-per-type 2
```

### 2. Assemble the measured and behavior releases

```bash
PYTHONPATH=src python scripts/assemble_proposal_evaluation_release.py \
  --measured-release evaluation/releases/egfr_her2_replay_doc3632549 \
  --behavior-release examples/evaluation_v075/behavior_release \
  --output-dir evaluation/releases/fos_eval_proposal_real_pilot_v075
```

### 3. Run baselines and heuristic Full FoS

```bash
PYTHONPATH=src python scripts/run_evaluation_suite_v2.py \
  --episodes evaluation/releases/fos_eval_proposal_real_pilot_v075/public/proposal_episodes.jsonl \
  --action-spaces evaluation/releases/fos_eval_proposal_real_pilot_v075/manifests/proposal_action_spaces.jsonl \
  --policies seed_only,greedy,tool_only,full_agent \
  --full-agent-backend heuristic \
  --output-dir outputs/proposal_eval_real_v075/runs
```

### 4. Generate proposal tables and reports

```bash
PYTHONPATH=src python scripts/generate_evaluation_report.py \
  --episodes evaluation/releases/fos_eval_proposal_real_pilot_v075/public/proposal_episodes.jsonl \
  --action-spaces evaluation/releases/fos_eval_proposal_real_pilot_v075/manifests/proposal_action_spaces.jsonl \
  --oracles evaluation/releases/fos_eval_proposal_real_pilot_v075/private_oracle/proposal_oracle.jsonl \
  --runs outputs/proposal_eval_real_v075/runs \
  --output-dir outputs/proposal_eval_real_v075/report \
  --pilot-label "FoS v0.7.5 proposal pilot" \
  --dataset-status "1 real measured episode + 12 controlled behavior fixtures; heuristic full-agent smoke test"
```

### 5. Run the Qwen/Ollama policy

Run this only when an OpenAI-compatible Ollama endpoint is available.

```bash
PYTHONPATH=src python scripts/run_evaluation_suite_v2.py \
  --episodes evaluation/releases/fos_eval_proposal_real_pilot_v075/public/proposal_episodes.jsonl \
  --action-spaces evaluation/releases/fos_eval_proposal_real_pilot_v075/manifests/proposal_action_spaces.jsonl \
  --policies full_agent \
  --full-agent-backend chat \
  --model qwen3:8b \
  --base-url http://127.0.0.1:11434/v1 \
  --output-dir outputs/proposal_eval_qwen_v075/runs
```

---

## Repository structure

```text
FoS/
├── src/
│   ├── stage_a/              # Evidence construction and local graph
│   ├── stage_b/              # Agent policy, tools, state machine, Critic
│   ├── stage_c/              # Conservative validation and reranking
│   └── evaluation/           # Schemas, runners, metrics, calibration
├── scripts/                  # Live runs, cache tools, evaluation pipelines
├── evaluation/
│   ├── datasets/
│   └── releases/
├── examples/                 # Fixtures and portable examples
├── configs/                  # Runtime and threshold configuration
├── docs/
│   ├── evaluation/
│   ├── proposal/
│   └── archive/versioned/
├── notebooks/
├── tests/
├── FoS_v075_proposal_evaluation_colab.ipynb
└── pyproject.toml
```

---

## Main output artifacts

A run can produce:

- Stage A context and target-pair evidence cache
- candidate-centered local graphs
- Stage B result JSON
- accepted, deferred, and rejected candidate records
- full optimization trajectory
- Stage C result JSON
- Stage C Markdown report
- evaluation run manifests
- measured and behavior result tables
- dataset card and method description
- leakage audit and checksum manifests
- policy calibration outputs

---

## Scientific and implementation boundaries

The current repository does **not** claim that:

- one measured episode establishes general performance,
- the Neighbor-KNN surrogate is independent validation,
- docking alone validates selectivity,
- controlled behavior fixtures represent biological efficacy,
- heuristic Full FoS performance demonstrates an LLM advantage.

The following work remains for a full scientific benchmark:

- repeated Qwen/API full-agent evaluation,
- at least two additional selectivity target pairs,
- frozen validation and hidden-test releases,
- independent Stage C predictor and docking providers,
- confidence calibration,
- no-Critic, no-Reflection, and no-Stage-C ablations.

---

## Documentation

Key documents:

- `docs/IMPLEMENTATION_STATUS.md`
- `docs/evaluation/EVALUATION_BENCHMARK_V074.md`
- `docs/evaluation/EVALUATION_GUIDE.md`
- `docs/proposal/PROPOSAL_SECTION4.md`
- `STAGE_B_SCHEMA.md`
- `STAGE_C_SCHEMA.md`
- `TRAJECTORY_MODE.md`
- `PROVISIONAL_TRAJECTORY_POLICY.md`
- `FoS_v075_proposal_evaluation_colab.ipynb`

---

## Team

**FoS — Future of Systems Biology and Medicine Laboratory**

Agent name: **Findings of Selectivity**
