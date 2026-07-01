"""
Smoke-test the custom Lucerne CityLearn dataset.

Two stages:
  1. Pre-flight checks (no CityLearn needed): confirm every CSV the schema
     references exists, loads, has the expected columns, and that all the
     time-series files share the same row count as the simulation horizon.
     This catches the most common "my dataset is broken" problems with a
     clear message before CityLearn is even imported.
  2. CityLearn baseline run, following the QuickStart "No Control (Baseline)"
     example: load the schema, step a BaselineAgent through one episode, and
     print the KPI table. If this completes, the dataset is structurally
     compatible with CityLearn.

Run from the directory that contains `luzern_citylearn/schema.json`
(i.e. the parent of the luzern_citylearn folder), or edit SCHEMA below.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SCHEMA = "luzern_citylearn/schema.json"

WEATHER_COLS = [
    "outdoor_dry_bulb_temperature",
    "outdoor_relative_humidity",
    "diffuse_solar_irradiance",
    "direct_solar_irradiance",
]
WEATHER_COLS += [f"{c}_predicted_{h}" for c in WEATHER_COLS for h in (1, 2, 3)]
PRICING_COLS = [
    "electricity_pricing",
    "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2",
    "electricity_pricing_predicted_3",
]
CARBON_COLS = ["carbon_intensity"]
BUILDING_REQUIRED = ["month", "hour", "day_type", "solar_generation", "non_shiftable_load"]


# ----------------------------------------------------------------------
# Stage 1 — pre-flight checks
# ----------------------------------------------------------------------
def preflight(schema_path: str) -> int:
    schema_path = Path(schema_path)
    root = schema_path.parent
    print(f"== Pre-flight checks ({schema_path}) ==")

    with open(schema_path) as f:
        schema = json.load(f)

    start = schema["simulation_start_time_step"]
    end = schema["simulation_end_time_step"]
    horizon = end - start + 1
    print(f"simulation horizon: steps {start}..{end}  ->  {horizon} time steps")

    problems = []

    def check_csv(name, expected_cols, expect_rows=True):
        path = root / name
        if not path.exists():
            problems.append(f"MISSING file: {path}")
            return None
        df = pd.read_csv(path)
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            problems.append(f"{name}: missing columns {missing}")
        if expect_rows and len(df) < horizon:
            problems.append(
                f"{name}: has {len(df)} rows but horizon needs >= {horizon}"
            )
        print(f"  {name}: {len(df)} rows, {len(df.columns)} cols")
        return df

    # Shared files (referenced by every building; check once via Building_1)
    buildings = schema["buildings"]
    first = next(iter(buildings.values()))
    check_csv(first["weather"], WEATHER_COLS)
    check_csv(first["pricing"], PRICING_COLS)
    check_csv(first["carbon_intensity"], CARBON_COLS)

    # Per-building energy files
    for bname, b in buildings.items():
        if not b.get("include", True):
            continue
        df = check_csv(b["energy_simulation"], BUILDING_REQUIRED)
        # warn (don't fail) if .pth dynamics model is referenced but absent
        dyn = b.get("dynamics", {}).get("attributes", {}).get("filename")
        if dyn and not (root / dyn).exists():
            problems.append(f"{bname}: dynamics model file missing: {root / dyn}")

    if problems:
        print("\nPRE-FLIGHT FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("Pre-flight OK: all referenced files present and consistent.\n")
    return 0


# ----------------------------------------------------------------------
# Stage 2 — CityLearn baseline run (QuickStart "No Control")
# ----------------------------------------------------------------------
def run_citylearn(schema_path: str) -> int:
    try:
        from citylearn.agents.base import BaselineAgent as Agent
        from citylearn.citylearn import CityLearnEnv
    except ImportError:
        print("CityLearn not installed. Install with:  pip install CityLearn")
        return 1

    print("== CityLearn baseline run ==")
    # central_agent=True keeps it simple (single controller, no per-building RL)
    env = CityLearnEnv(schema_path, central_agent=True)
    model = Agent(env)

    observations, _ = env.reset()
    while not env.terminated:
        actions = model.predict(observations)
        observations, reward, info, terminated, truncated = env.step(actions)

    kpis = env.evaluate()
    kpis = kpis.pivot(index="cost_function", columns="name", values="value").round(3)
    kpis = kpis.dropna(how="all")
    print("\nKPIs (1.0 = same as baseline / uncontrolled):")
    print(kpis.to_string())
    print("\nSUCCESS: the custom dataset ran through CityLearn end to end.")
    return 0


if __name__ == "__main__":
    schema = sys.argv[1] if len(sys.argv) > 1 else SCHEMA
    if preflight(schema) != 0:
        sys.exit(1)
    sys.exit(run_citylearn(schema))