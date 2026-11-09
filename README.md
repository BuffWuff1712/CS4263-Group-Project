# Maritime Transshipment Routing and Scheduling — Classical vs. LLM

This project compares two ways of solving the **Maritime Transshipment Routing and Scheduling Problem (MTRSP)**:

- **Classical:** a constraint satisfaction (CSP) solver with backtracking search (`csp_itinerary.py`).
- **LLM-based:** a **Plan–Simulate–Repair pipeline** in which an LLM writes and revises schedules (`pipeline.py`).

Both approaches are checked by the same deterministic simulator (`simulator.py`). The goal is a feasible vessel schedule that delivers every shipment before its deadline while satisfying routing, vessel capacity, cargo movement, and scheduling constraints.

## Project Components

| File | Responsibility |
|---|---|
| `pipeline.py` | Runs the Plan–Simulate–Repair loop and saves the full attempt history. Main entry point for the LLM approach. |
| `planner.py` | Connects to the selected provider (OpenAI, SOCLAAS, or OpenRouter) and asks the LLM for an initial voyage schedule. |
| `simulator.py` | Deterministically validates a schedule against constraints C1–C13 and reports violations, delivered quantities, and a movement trace. |
| `repair.py` | Sends the instance, previous schedule, and violations back to the LLM for a revised schedule. Retries on rate limits. |
| `feedback.py` | Turns simulator violations into grouped, prioritised feedback with verified facts for guided repair. |
| `safeguards.py` | Holds the safeguard configuration recorded in each results file. |
| `csp_itinerary.py` | Classical CSP solver: constraint propagation and backtracking search, with cargo-path preprocessing, grouped itinerary branching, optional restarts and a time limit. |
| `schemas.py` | Pydantic data models for voyages, cargo loads, schedules, violations, and validation reports. |
| `run_planner.py` | Runs only the planning step and saves the candidate schedule, without checking feasibility. |
| `run_qwen_quick.py` | One-shot baseline across several Qwen models and instances. The simulator scores each output, but no feedback is sent back. |
| `prompts/plan.txt` | The LLM's initial planning instructions. |
| `prompts/repair.txt` | The LLM's repair instructions. |
| `prompts/plan_outline.txt`, `prompts/repair_outline.txt` | Instructions for writing a plan outline with the schedule. |
| `prompts/repair_trace.txt` | Instructions for using the simulation trace during repair. |
| `data/*.json` | Problem instances (`easy`, `medium`, `hard`, `extra_hard`) with ports, routes, vessels, shipments, and constraints. |
| `tests/` | Pytest tests for the planner, simulator trace, and safeguard wiring (no API calls). |

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

## Safeguards

The three group flags can be tested separately or enabled together:

| Flag | What it does |
|---|---|
| `--planning-support` | Asks for a shipment-route and vessel outline before the schedule and carries it into repair. |
| `--guided-repair` | Gives repairs prioritised simulator violations, computed timing and capacity facts, and a movement trace. |
| `--repair-guards` | Recovers a complete schedule from malformed output when possible and rejects repairs that lose delivered units. |
| `--all-safeguards` | Enables the three groups together. |

`--tiered-safeguards` runs the latest staged workflow: planning support
on the initial plan, guided repair first, then an outline-assisted repair
if another attempt is needed. Output recovery can handle an initial or
repair reply; delivery checks compare proposed repairs with the accepted
schedule. The simulator alone decides validity, and the repair limit still
applies. The selected safeguards and attempt history are saved in the
result file.

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

# With the tiered safeguards, and a custom output file
python pipeline.py --provider soclaas --instance data/hard.json --tiered-safeguards \
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
| `--planning-support` | Rolling outline. |
| `--guided-repair` | Grounded feedback and simulation trace. |
| `--repair-guards` | Output recovery and delivery preservation. |
| `--all-safeguards` | All three groups. |
| `--tiered-safeguards` | Use the latest staged safeguard workflow. |
| `--initial-schedule PATH` | Start from a saved schedule or pipeline result instead of calling the planner, for controlled comparisons. |
| `--resume-final` | With `--initial-schedule`, use that result's final schedule instead of its first. |
| `--request-timeout SECONDS` | Timeout for each model request. |

The three group flags can be combined; `--tiered-safeguards` is a separate
workflow. `planner_metadata.safeguards` records the effective components:

```bash
python pipeline.py --provider soclaas --instance data/hard.json --all-safeguards
```

To compare the groups from the same starting schedule:

```bash
python pipeline.py --provider soclaas --output results/base.json
python pipeline.py --provider soclaas --initial-schedule results/base.json \
    --output results/group_baseline.json
python pipeline.py --provider soclaas --initial-schedule results/base.json \
    --planning-support --output results/group_planning.json
python pipeline.py --provider soclaas --initial-schedule results/base.json \
    --guided-repair --output results/group_guided.json
python pipeline.py --provider soclaas --initial-schedule results/base.json \
    --repair-guards --output results/group_guards.json
python pipeline.py --provider soclaas --initial-schedule results/base.json \
    --all-safeguards --output results/group_all.json
```

Use the same model, instance, reasoning level, and repair budget in every
run. An initial schedule contains no outline, so the first repair in the
planning-support group creates one.

## Running the Classical CSP Baseline

```bash
python csp_itinerary.py                          # easy, medium and hard
python csp_itinerary.py hard --time-limit 600
python csp_itinerary.py hard --baseline          # plain CSP branching, for comparison
```

`--time-limit` is in seconds per instance (0 means no limit, the default). `--restart-base` sets the number of nodes per Luby restart unit (0 means no restarts). Run `python csp_itinerary.py --help` for the switches that turn each optimisation on or off. Instances with chokepoints, mandatory stops, non-zero handling time, transfer delays or limited port storage are reported as `unsupported`. Schedules and metrics are saved to `results/classical_csp_itinerary/`.

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
