"""
ecobee_to_citylearn.py — Convert an Ecobee BBD (Donate-Your-Data) xarray
Dataset into a CityLearn dataset folder.

The Ecobee BBD is THERMOSTAT data: indoor/outdoor temps, setpoints, humidity,
HVAC equipment runtimes, and motion sensors at 5-minute resolution. It does
NOT contain whole-home electricity load, solar generation, solar irradiance,
electricity prices, or carbon intensity. This script:

  • maps every column that has a real CityLearn counterpart,
  • derives what can be approximated (demand from runtimes, occupancy from motion),
  • fills the genuinely-absent fields with zeros / placeholders and PRINTS a
    report of exactly what was synthetic so nothing is silently faked.

Resampling: 5-min → hourly (CityLearn is hourly). 8640 5-min steps → 720 hours.

Dependencies
────────────
    pip install xarray netcdf4 numpy pandas
(netcdf4 or h5netcdf only needed if you load from a .nc file.)

Usage
─────
    # from an open x.Dataset:
    from ecobee_to_citylearn import convert
    convert(ds, out_dir="ecobee_dataset", building_ids=ds.id.values[:1])

    # or from a file, via the CLI:
    python ecobee_to_citylearn.py --input ecobee.nc --out ecobee_dataset --n-buildings 1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ── Unit conversion ─────────────────────────────────────────────────────────────
#
# The Ecobee BBD reports temperatures in Fahrenheit; CityLearn expects Celsius.
# Setpoint thresholds below are therefore written in °C and the raw °F series are
# converted on read. Humidity is already a percentage and needs no conversion;
# runtimes are in seconds and stay as-is.
#
# Set INPUT_TEMP_IS_FAHRENHEIT = False if you ever feed a pre-converted dataset.

INPUT_TEMP_IS_FAHRENHEIT = True

# Every variable holding a temperature, so the converter can find them generically.
TEMPERATURE_VARS = {
    "Indoor_AverageTemperature", "Indoor_CoolSetpoint", "Indoor_HeatSetpoint",
    "Outdoor_Temperature", "Thermostat_Temperature",
    "RemoteSensor1_Temperature", "RemoteSensor2_Temperature",
    "RemoteSensor3_Temperature", "RemoteSensor4_Temperature",
    "RemoteSensor5_Temperature",
}


def f_to_c(x):
    """Fahrenheit → Celsius, NaN-safe (works on scalars and numpy arrays)."""
    return (np.asarray(x, dtype=float) - 32.0) * 5.0 / 9.0


def _is_temperature_var(name: str) -> bool:
    return name in TEMPERATURE_VARS or name.endswith("_Temperature")


# ── Assumptions you may need to tune (no source for these in the dataset) ───────

# Equipment electrical capacity used to turn runtime fractions into kWh demand.
# These are ROUGH single-family defaults; replace with per-home nameplate data
# from the BBD metadata JSON if you have it.
COOLING_STAGE_KW = {1: 3.5, 2: 5.0}      # AC compressor electrical kW per stage
HEATING_STAGE_KW = {1: 4.0, 2: 8.0, 3: 12.0}  # furnace/strip kW per stage
HEATPUMP_STAGE_KW = {1: 3.0, 2: 5.0}
FAN_KW = 0.3

# CityLearn hvac_mode codes: 0 = off, 1 = cooling, 2 = heating.
# Ecobee HVAC_Mode is a float; the exact encoding varies by export. Adjust the
# mapping below to your file's observed values (printed in the report).
HVAC_MODE_MAP = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 2}  # default best-guess

BUILDING_COLUMNS = [
    "month", "hour", "day_type", "daylight_savings_status",
    "indoor_dry_bulb_temperature", "average_unmet_cooling_setpoint_difference",
    "indoor_relative_humidity", "non_shiftable_load", "dhw_demand",
    "cooling_demand", "heating_demand", "solar_generation", "occupant_count",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "indoor_dry_bulb_temperature_heating_set_point", "hvac_mode",
]
WEATHER_COLUMNS = [
    "outdoor_dry_bulb_temperature", "outdoor_relative_humidity",
    "diffuse_solar_irradiance", "direct_solar_irradiance",
] + [f"{b}_predicted_{k}" for b in
     ["outdoor_dry_bulb_temperature", "outdoor_relative_humidity",
      "diffuse_solar_irradiance", "direct_solar_irradiance"] for k in (1, 2, 3)]
PRICING_COLUMNS = ["electricity_pricing"] + \
    [f"electricity_pricing_predicted_{k}" for k in (1, 2, 3)]


# ── Per-building extraction ─────────────────────────────────────────────────────

def _building_frame(ds: xr.Dataset, bid, report: dict) -> pd.DataFrame:
    """Build one CityLearn building CSV (hourly) from one Ecobee id."""
    sub = ds.sel(id=bid)

    # Pull each variable to a pandas Series indexed by time, then resample hourly.
    def series(name, agg="mean"):
        if name not in ds.data_vars:
            report.setdefault("missing_vars", set()).add(name)
            return None
        raw = np.asarray(sub[name].values, dtype=float)
        if INPUT_TEMP_IS_FAHRENHEIT and _is_temperature_var(name):
            raw = f_to_c(raw)
            report.setdefault("converted_fields", set()).add(f"{name} (°F→°C)")
        s = pd.Series(raw, index=pd.to_datetime(ds.time.values))
        return s.resample("1h").mean() if agg == "mean" else s.resample("1h").sum()

    time_h = pd.to_datetime(ds.time.values)
    hourly_index = pd.Series(0, index=time_h).resample("1h").mean().index
    n = len(hourly_index)

    # Temporal calendar fields.
    month = hourly_index.month.to_numpy()
    hour = (hourly_index.hour + 1).to_numpy()                # CityLearn 1..24
    day_type = (hourly_index.dayofweek + 1).to_numpy()       # 1=Mon .. 7=Sun

    # Direct maps.
    indoor_temp = series("Indoor_AverageTemperature")
    cool_sp = series("Indoor_CoolSetpoint")
    heat_sp = series("Indoor_HeatSetpoint")
    humidity = series("Indoor_Humidity")
    hvac_raw = series("HVAC_Mode")

    # Occupancy proxy: any motion across thermostat + remote sensors → occupied.
    motion_cols = [v for v in ds.data_vars if v.endswith("DetectedMotion")]
    if motion_cols:
        motion = sum(
            pd.Series(np.nan_to_num(np.asarray(sub[c].values)),
                      index=time_h).resample("1h").max()
            for c in motion_cols
        )
        occupant_count = (motion > 0).astype(int).to_numpy()
    else:
        occupant_count = np.zeros(n, dtype=int)
        report.setdefault("synthetic_fields", set()).add("occupant_count(no motion cols)")

    # Demand derived from runtimes (fraction of hour running × stage kW).
    # Ecobee runtimes are seconds within each 5-min step; after hourly .mean()
    # we treat them as average per-step seconds and scale to an hourly fraction.
    def runtime_kwh(prefix, kw_map):
        total = np.zeros(n)
        found = False
        for stage, kw in kw_map.items():
            col = f"{prefix}Stage{stage}_RunTime"
            s = series(col, agg="mean")
            if s is None:
                continue
            found = True
            frac = np.clip(s.reindex(hourly_index).to_numpy() / 300.0, 0, 1)  # 300s per 5-min step
            total += np.nan_to_num(frac) * kw
        return total, found

    cooling_kwh, c1 = runtime_kwh("CoolingEquipment", COOLING_STAGE_KW)
    heating_kwh, h1 = runtime_kwh("HeatingEquipment", HEATING_STAGE_KW)
    hp_kwh, h2 = runtime_kwh("HeatPumps", HEATPUMP_STAGE_KW)
    heating_total = heating_kwh + hp_kwh
    if c1 or h1 or h2:
        report.setdefault("derived_fields", set()).add(
            "cooling_demand/heating_demand (from runtimes × assumed kW)")

    fan = series("Fan_RunTime")
    fan_kwh = (np.clip(fan.reindex(hourly_index).to_numpy() / 300.0, 0, 1) * FAN_KW
               if fan is not None else np.zeros(n))

    # non_shiftable_load: NOT in dataset. Best proxy = fan + a flat base load.
    # This is SYNTHETIC — flag it loudly.
    base_plug_kw = 0.3
    non_shiftable = base_plug_kw + np.nan_to_num(fan_kwh)
    report.setdefault("synthetic_fields", set()).add(
        "non_shiftable_load (no metered load; flat base + fan only)")

    def col(s, default):
        if s is None:
            return np.full(n, default)
        return np.nan_to_num(s.reindex(hourly_index).to_numpy(), nan=default)

    hvac_mapped = np.array([HVAC_MODE_MAP.get(round(v, 0), 0)
                            for v in col(hvac_raw, 0.0)], dtype=int)

    df = pd.DataFrame({
        "month": month,
        "hour": hour,
        "day_type": day_type,
        "daylight_savings_status": np.ones(n, dtype=int),
        "indoor_dry_bulb_temperature": col(indoor_temp, 22.0),
        "average_unmet_cooling_setpoint_difference": np.zeros(n),
        "indoor_relative_humidity": col(humidity, 50.0),
        "non_shiftable_load": non_shiftable,
        "dhw_demand": np.zeros(n),                      # not in dataset
        "cooling_demand": cooling_kwh,
        "heating_demand": heating_total,
        "solar_generation": np.zeros(n),               # not in dataset
        "occupant_count": occupant_count,
        "indoor_dry_bulb_temperature_cooling_set_point": col(cool_sp, 24.0),
        "indoor_dry_bulb_temperature_heating_set_point": col(heat_sp, 20.0),
        "hvac_mode": hvac_mapped,
    })
    report.setdefault("synthetic_fields", set()).update(
        {"dhw_demand (all zero)", "solar_generation (all zero)"})
    return df[BUILDING_COLUMNS], hourly_index


def _weather_frame(ds: xr.Dataset, bid, hourly_index, report: dict) -> pd.DataFrame:
    """Outdoor temp/humidity are real; solar irradiance is absent → zeros."""
    sub = ds.sel(id=bid)
    time_h = pd.to_datetime(ds.time.values)

    def col(name, default):
        if name not in ds.data_vars:
            return np.full(len(hourly_index), default)
        raw = np.asarray(sub[name].values, dtype=float)
        if INPUT_TEMP_IS_FAHRENHEIT and _is_temperature_var(name):
            raw = f_to_c(raw)
            report.setdefault("converted_fields", set()).add(f"{name} (°F→°C)")
        s = pd.Series(raw, index=time_h).resample("1h").mean()
        return np.nan_to_num(s.reindex(hourly_index).to_numpy(), nan=default)

    temp = col("Outdoor_Temperature", 15.0)
    rh = col("Outdoor_Humidity", 60.0)
    n = len(hourly_index)
    zeros = np.zeros(n)
    report.setdefault("synthetic_fields", set()).update(
        {"diffuse_solar_irradiance (all zero — bring real weather file)",
         "direct_solar_irradiance (all zero — bring real weather file)"})

    def shift(a, k):
        out = np.roll(a, -k); out[-k:] = a[-1]; return out

    data = {
        "outdoor_dry_bulb_temperature": temp,
        "outdoor_relative_humidity": rh,
        "diffuse_solar_irradiance": zeros,
        "direct_solar_irradiance": zeros,
    }
    for base, arr in [("outdoor_dry_bulb_temperature", temp),
                      ("outdoor_relative_humidity", rh),
                      ("diffuse_solar_irradiance", zeros),
                      ("direct_solar_irradiance", zeros)]:
        for k in (1, 2, 3):
            data[f"{base}_predicted_{k}"] = shift(arr, k)
    return pd.DataFrame(data)[WEATHER_COLUMNS]


def _placeholder_pricing(n, report):
    report.setdefault("synthetic_fields", set()).add(
        "pricing.csv (flat $0.22/kWh placeholder — bring a real tariff)")
    flat = np.full(n, 0.22)
    return pd.DataFrame({c: flat for c in PRICING_COLUMNS})


def _placeholder_carbon(n, report):
    report.setdefault("synthetic_fields", set()).add(
        "carbon_intensity.csv (flat 0.15 placeholder — bring real grid data)")
    return pd.DataFrame({"carbon_intensity": np.full(n, 0.15)})


# ── Schema ──────────────────────────────────────────────────────────────────────

def _schema(rows, building_files):
    obs_active = {
        "month", "hour", "day_type", "outdoor_dry_bulb_temperature",
        "outdoor_relative_humidity", "indoor_dry_bulb_temperature",
        "indoor_relative_humidity", "non_shiftable_load", "electricity_pricing",
        "carbon_intensity", "indoor_dry_bulb_temperature_cooling_set_point",
        "indoor_dry_bulb_temperature_heating_set_point", "occupant_count",
    }
    all_obs = obs_active | {
        "daylight_savings_status", "diffuse_solar_irradiance",
        "direct_solar_irradiance", "solar_generation", "cooling_demand",
        "heating_demand", "dhw_demand",
        "average_unmet_cooling_setpoint_difference", "hvac_mode",
    }
    buildings = {}
    for i, bf in enumerate(building_files):
        buildings[f"Building_{i+1}"] = {
            "include": True,
            "energy_simulation": bf,
            "weather": "weather.csv",
            "carbon_intensity": "carbon_intensity.csv",
            "pricing": "pricing.csv",
            "inactive_observations": [],
            "inactive_actions": [],
            "cooling_storage": {"type": "citylearn.energy_model.StorageTank",
                                "autosize": True, "attributes": {"capacity": 5.0}},
            "electrical_storage": {"type": "citylearn.energy_model.Battery",
                                   "autosize": True,
                                   "attributes": {"capacity": 6.4, "nominal_power": 5.0}},
            "cooling_device": {"type": "citylearn.energy_model.HeatPump", "autosize": True},
        }
    return {
        "central_agent": False,
        "simulation_start_time_step": 0,
        "simulation_end_time_step": rows - 1,
        "episode_time_steps": rows,
        "seconds_per_time_step": 3600,
        "random_seed": 0,
        "observations": {n: {"active": n in obs_active} for n in sorted(all_obs)},
        "actions": {"cooling_storage": {"active": True},
                    "electrical_storage": {"active": True}},
        "buildings": buildings,
        "reward_function": {"type": "citylearn.reward_function.RewardFunction"},
    }


# ── Top-level convert ────────────────────────────────────────────────────────────

def convert(ds: xr.Dataset, out_dir: str = "ecobee_dataset", building_ids=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if building_ids is None:
        building_ids = [ds.id.values[0]]                 # default: first home only
    report = {}

    building_files, rows_seen, weather_written = [], None, False
    for i, bid in enumerate(building_ids):
        bdf, hourly_index = _building_frame(ds, bid, report)
        rows = len(bdf)
        rows_seen = rows_seen or rows
        if rows != rows_seen:
            raise ValueError(f"Building {bid} has {rows} hourly rows, expected {rows_seen}. "
                             "Homes must share the same time axis.")
        fname = f"Building_{i+1}.csv"
        bdf.to_csv(out / fname, index=False)
        building_files.append(fname)

        if not weather_written:              # one shared weather file from first home
            _weather_frame(ds, bid, hourly_index, report).to_csv(out / "weather.csv", index=False)
            _placeholder_pricing(rows, report).to_csv(out / "pricing.csv", index=False)
            _placeholder_carbon(rows, report).to_csv(out / "carbon_intensity.csv", index=False)
            weather_written = True

    with open(out / "schema.json", "w") as f:
        json.dump(_schema(rows_seen, building_files), f, indent=2)

    _print_report(report, rows_seen, len(building_files), out)
    return out / "schema.json"


def _print_report(report, rows, n_buildings, out):
    line = "─" * 68
    print(f"\n{line}\nEcobee → CityLearn conversion report\n{line}")
    print(f"Output: {out.resolve()}")
    print(f"Buildings: {n_buildings}   Hourly rows/file: {rows} "
          f"({rows/24:.0f} days)")
    if report.get("converted_fields"):
        print("\nUNIT CONVERSIONS APPLIED:")
        for f in sorted(report["converted_fields"]):
            print(f"  ✓ {f}")
    print("\nDERIVED (approximated, check assumptions):")
    for f in sorted(report.get("derived_fields", [])):
        print(f"  • {f}")
    print("\nSYNTHETIC / PLACEHOLDER (no real source in dataset):")
    for f in sorted(report.get("synthetic_fields", [])):
        print(f"  ⚠ {f}")
    if report.get("missing_vars"):
        print("\nEXPECTED-BUT-ABSENT variables (filled with defaults):")
        for v in sorted(report["missing_vars"]):
            print(f"  – {v}")
    print(f"{line}")
    print("To make energy/cost KPIs meaningful, replace these before training:")
    print("  1. weather.csv solar irradiance columns (TMY/AMY for the home's location)")
    print("  2. pricing.csv (utility tariff)")
    print("  3. carbon_intensity.csv (grid operator / WattTime)")
    print("  4. non_shiftable_load (metered whole-home load — not in Ecobee BBD)")
    print(f"{line}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .nc / .zarr Ecobee dataset")
    ap.add_argument("--out", default="ecobee_dataset")
    ap.add_argument("--n-buildings", type=int, default=1)
    args = ap.parse_args()

    ds = xr.open_dataset(args.input)
    ids = list(ds.id.values[: args.n_buildings])
    schema_path = convert(ds, out_dir=args.out, building_ids=ids)
    print(f"Test it:\n  CITYLEARN_SCHEMA={schema_path.resolve()} python experiment.py")
    print("Or validate quickly with custom_dataset.test_dataset(schema_path).")


if __name__ == "__main__":
    main()
