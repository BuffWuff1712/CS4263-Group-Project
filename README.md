# Maritime Transshipment Routing and Scheduling — Solution B

This project implements an **LLM-based Plan–Simulate–Repair pipeline** to solve the Maritime Transshipment Routing and Scheduling Problem (MTRSP).

The objective is to generate a feasible vessel schedule that delivers all shipments before their deadlines while satisfying routing, vessel capacity, cargo movement, and scheduling constraints.

## Project Components

| File | Responsibility |
|---|---|
| `pipeline.py` | Coordinates the planning, simulation, and repair process. |
| `planner.py` | Uses Qwen/GPT-6.1 via the SOCLAAS API/OPENAI API to generate an initial voyage schedule from a problem instance. |
| `simulator.py` | Deterministically validates the proposed schedule against the problem constraints and reports violations. |
| `repair.py` | Uses Qwen/GPT-6.1 to revise an invalid schedule based on the simulator's feedback. |
| `schemas.py` | Defines Pydantic data models for voyages, cargo loads, schedules, violations, and validation reports. |
| `prompts/plan.txt` | Contains the LLM's initial planning instructions. |
| `prompts/repair.txt` | Contains the LLM's repair instructions. |
| `data/*.json` | Contains problem instances with ports, routes, vessels, shipments, and constraints. |

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
                  +-- Repeat up to MAX_REPAIRS
```

### 1. Plan

`planner.py` reads the problem instance and `prompts/plan.txt`, then calls `qwen3.6:35b` to generate a complete voyage schedule.

### 2. Simulate

`simulator.py` checks the schedule for constraint violations, including:

- Route validity, vessel compatibility, and voyage continuity.
- Vessel availability, voyage limits, and cargo capacity.
- Cargo conservation, splitting restrictions, and valid transshipments.
- Shipment deadlines, mandatory stops, and transport-leg limits.
- Chokepoint restrictions, where applicable.

It returns a `ValidationReport` containing the validity status, violations, and delivered quantities.

### 3. Repair

If the schedule is invalid, `repair.py` sends the original problem instance, previous schedule, and detected violations back to Qwen, using `prompts/repair.txt`.

Qwen generates a complete revised schedule, which is passed back to the simulator for validation.

### 4. Termination

The pipeline terminates when either:

- A valid schedule is found.
- The maximum number of repair attempts is reached.
- A planning, parsing, or API error prevents continuation.

Results and attempt histories are saved to `results/pipeline_result.json`.

## Running the Project

Configure `.env` with the SOCLAAS API credentials and model settings.

```ini
SOCLAAS_API_KEY=your_api_key
SOCLAAS_BASE_URL=your_base_url
SOCLAAS_MODEL=qwen3.6:35b

OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_API_KEY=your_secret_key
OPENAI_MODEL='gpt-6.1-sol'
```

Select the problem instance in `pipeline.py`:

```python
instance_path = BASE_DIR / "data" / "easy.json"
```

Run:

```bash
python pipeline.py
```

To evaluate another instance, change `easy.json` to `medium.json` or `hard.json`.

## Architecture

The **LLM is responsible for generating and revising schedules**, while the Python simulator independently verifies their feasibility.

The simulator does not perform constraint search, automatically correct schedules, or generate solutions.