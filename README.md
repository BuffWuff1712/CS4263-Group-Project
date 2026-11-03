# Maritime Transshipment Routing and Scheduling — Classical vs. LLM

This project compares two ways of solving the **Maritime Transshipment Routing and Scheduling Problem (MTRSP)**:

- **Classical:** a constraint satisfaction (CSP) solver with backtracking search (`csp.py`).
- **LLM-based:** a **Plan–Simulate–Repair pipeline** in which an LLM writes and revises schedules (`pipeline.py`).

Both approaches are checked by the same deterministic simulator (`simulator.py`). The goal is a feasible vessel schedule that delivers every shipment before its deadline while satisfying routing, vessel capacity, cargo movement, and scheduling constraints.

## Project Components

| File | Responsibility |
|---|---|
| `pipeline.py` | Runs the Plan–Simulate–Repair loop and saves the full attempt history. Main entry point for the LLM approach. |
| `planner.py` | Connects to the selected provider (OpenAI, SOCLAAS, or OpenRouter) and asks the LLM for an initial voyage schedule. |
| `simulator.py` | Deterministically validates a schedule against constraints C1–C13 and reports violations and delivered quantities. |
| `repair.py` | Sends the instance, previous schedule, and violations back to the LLM for a revised schedule. Retries on rate limits. |
| `feedback.py` | Turns simulator violations into grouped, prioritised feedback with verified facts (used with `--safeguard`). |
| `csp.py` | Classical CSP baseline: constraint propagation and backtracking search, with optional restarts and a time limit. |
| `schemas.py` | Pydantic data models for voyages, cargo loads, schedules, violations, and validation reports. |
| `run_planner.py` | Runs only the planning step and saves the candidate schedule, without checking feasibility. |
| `run_qwen_quick.py` | One-shot baseline across several Qwen models and instances. The simulator scores each output, but no feedback is sent back. |
| `prompts/plan.txt` | The LLM's initial planning instructions. |
| `prompts/repair.txt` | The LLM's repair instructions. |
| `data/*.json` | Problem instances (`easy`, `medium`, `hard`) with ports, routes, vessels, shipments, and constraints. |
| `tests/` | Pytest tests. |

## Pipeline Workflow

```text
Problem Instance (JSON)
          |
          v
      planner.py
   Generate initial plan
          |
          v
      schemas.py
   Validate JSON structure
          |
          v
      simulator.py
   Check CSP constraints
          |
          v
     Valid schedule?
       /       \
      Yes       No
       |         |
       v         v
    Success   repair.py
              LLM revises plan
                  |
                  v
              simulator.py
              Revalidate
                  |
                  +-- Repeat up to --max-repairs
```

### 1. Plan

`planner.py` reads the problem instance and `prompts/plan.txt`, then asks the selected model to generate a complete voyage schedule.

### 2. Simulate

`simulator.py` checks the schedule for constraint violations, including:

- Route validity, vessel compatibility, and voyage continuity.
- Vessel availability, voyage limits, and cargo capacity.
- Cargo conservation, splitting restrictions, and valid transshipments.
- Shipment deadlines, mandatory stops, and transport-leg limits.
- Chokepoint restrictions, where applicable.

It returns a `ValidationReport` containing the validity status, violations, and delivered quantities.

### 3. Repair

If the schedule is invalid, `repair.py` sends the problem instance, previous schedule, and detected violations back to the model, using `prompts/repair.txt`. The model returns a complete revised schedule, which goes back to the simulator.

### 4. Termination

The pipeline stops when:

- A valid schedule is found.
- The maximum number of repair attempts is reached (default 3).
- A planning, parsing, or API error prevents continuation.

Results and attempt histories are saved to the path given by `--output` (default `results/pipeline_result.json`).

## Safeguard Mode (`--safeguard`)

Off by default. With `--safeguard`, `pipeline.py` adds three safeguards to the repair loop. The simulator still decides whether a schedule is valid.

1. **Grounded feedback** (`feedback.py`): violations are grouped by constraint and ordered so structural and cargo-movement errors come before delivery errors. Each violation includes facts computed from the instance and schedule, such as excess units over capacity or hours past a deadline. Repeated unit-level delivery messages are shortened, but every shipment shortfall is listed.
2. **Output recovery** (`repair.py`): if the repair reply is not clean schedule JSON, complete schedules embedded in the text are extracted, simulated, and the one with the fewest violations is used. If none is found, the model gets one retry with a stricter format prompt.
3. **Delivery preservation** (`pipeline.py`): a repaired schedule that is still invalid and delivers fewer units of any shipment than before is rejected. The previous schedule is kept, and the next repair prompt lists the lost shipments. A rejected repair still counts as an attempt.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` for the provider you will use:

```ini
SOCLAAS_BASE_URL="https://soclaas-api.comp.nus.edu.sg/v1"
SOCLAAS_API_KEY=""
SOCLAAS_MODEL='qwen3.6:35b'
SOCLAAS_REASONING_EFFORT=medium

OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_API_KEY=""
OPENAI_MODEL='gpt-6.1-sol'

# Optional
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
OPENROUTER_API_KEY=""
OPENROUTER_MODEL='qwen/qwen3.8-max-prime'
```

`.env` is ignored by git. Never commit API keys.

## Running the LLM Pipeline

```bash
# Defaults: --provider openai, --instance data/hard.json
python pipeline.py

# Choose a provider and instance
python pipeline.py --provider soclaas --instance data/easy.json

# With safeguards, and a custom output file
python pipeline.py --provider soclaas --instance data/hard.json --safeguard \
    --output results/hard_safeguard.json
```

Options:

| Flag | Meaning |
|---|---|
| `--provider {openai,soclaas,openrouter}` | API provider (default `openai`). |
| `--model MODEL` | Override the model set in `.env`. |
| `--instance PATH` | Problem instance (default `data/hard.json`). |
| `--output PATH` | Where to save results (default `results/pipeline_result.json`). |
| `--max-repairs N` | Maximum repair attempts (default 3). |
| `--reasoning-effort LEVEL` | Only for models that support it. Otherwise read from `.env`. |
| `--safeguard` | Turn on the safeguards described above. |
| `--initial-schedule PATH` | Start from a saved schedule or pipeline result instead of calling the planner, for controlled comparisons. |
| `--resume-final` | With `--initial-schedule`, use that result's final schedule instead of its first. |
| `--request-timeout SECONDS` | Timeout for each model request. |

To compare runs with and without safeguards from the same starting schedule:

```bash
python pipeline.py --provider soclaas --output results/base.json
python pipeline.py --provider soclaas --initial-schedule results/base.json --safeguard \
    --output results/safeguard.json
```

## Running the Classical CSP Baseline

```bash
python csp.py                          # easy, medium and hard
python csp.py easy medium --time-limit 600 --restart-base 0
```

`--time-limit` is in seconds per instance (0 means no limit). `--restart-base` sets the number of nodes per Luby restart unit (0 means no restarts). Schedules and metrics are saved to `results/classical_csp/`.

## Other Scripts

```bash
# Planning step only (no simulation or repair)
python run_planner.py --provider soclaas --instance data/easy.json

# One-shot Qwen baseline across instances
python run_qwen_quick.py --provider soclaas --instances easy medium hard
```

## Tests

```bash
pytest
```

## Architecture

The **LLM is responsible for generating and revising schedules**, while the Python simulator independently verifies their feasibility. The simulator does not search, correct schedules, or generate solutions.

The **CSP solver** searches for a feasible schedule directly, and its output is checked by the same simulator, so both approaches are judged on the same constraints.
