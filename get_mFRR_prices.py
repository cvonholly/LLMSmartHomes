"""
flex_value_mfrr.py
==================
Estimate the yearly value of electricity flexibility in Switzerland from the
tertiary (mFRR) balancing market.


DATA
------
Downloaded from https://www.swissgrid.ch/de/home/operation/grid-data/control-energy-system-balance.html (bottom of page)

METHOD
------
mFRR (manual Frequency Restoration Reserve) pays an *energy* price (EUR/MWh)
for energy delivered ONLY in the quarter-hours where the reserve is actually
activated. A flexible load that REDUCES consumption provides UPWARD regulation
(mFRR+), so the relevant series is the mFRR+ activated-energy average price
(column P in the Swissgrid file, EUR/MWh).

Your proposed formula:
        annual_value = flex_potential[kWh/yr] x average_price[EUR/kWh]
is structurally correct. The only subtlety is which "average price" to use:

  - A price-taker who always bids captures the mean price OVER ACTIVATED
    intervals (you are only paid when called) -> "always-bid capture price".
  - A selective bidder who only offers in the most expensive intervals
    captures a higher price but on fewer hours -> "selective capture price".

Because your kWh/yr potentials already encode *how much energy* each device
can shift over the year, the cleanest estimate is:

        annual_value[EUR] = flex_kWh_per_yr x capture_price[EUR/kWh]

This script computes several capture-price scenarios (annual, day/night,
seasonal, selective) from the data and applies them to each device profile.

Note: the file covers Jan-Jun 2026 (~6 months). Prices are annualised by using
the period mean as the best estimate of the full-year average (stated below).

Usage:
    python flex_value_mfrr.py --csv Ausgleichsenergie-und-Regelenergie-2026.csv
"""

import argparse
import pandas as pd
import numpy as np

# --- Device flexibility profiles (from user) -------------------------------
# nominal_kW is informational; the kWh/yr columns drive the energy value.
DEVICES = [
    # name,                         nominal_kW, standard_kWh, tolerant_kWh, season
    ("Heat pump (heating season)",   1.00,  400,  800, "heating"),
    ("EV (smart charging)",         11.00,  989, 1380, "all"),
    ("Home battery (daily cycle)",  10.00, 2970, 2970, "all"),
    ("Air conditioning (cooling)",   0.50,   70,  140, "cooling"),
    # Solar PV is generation, not a controllable load reduction -> excluded
    # from mFRR+ value here (it can provide mFRR- / downward, different price).
]


def load_prices(csv_path):
    df = pd.read_csv(csv_path, sep=";", decimal=".")
    df.columns = [c.split(":")[0] if ":" in c else c for c in df.columns]
    df["dt"] = pd.to_datetime(df.iloc[:, 0], format="%d.%m.%Y %H:%M")
    df["P"] = pd.to_numeric(df["P"], errors="coerce")   # mFRR+ activated EUR/MWh
    df["hour"] = df["dt"].dt.hour
    df["month"] = df["dt"].dt.month
    return df


def capture_prices(df):
    """Return a dict of capture prices in EUR/MWh (paid only when activated)."""
    act = df[df["P"].notna()]
    day = act[(act.hour >= 8) & (act.hour < 20)]
    night = act[(act.hour < 8) | (act.hour >= 20)]
    heating = act[act.month.isin([1, 2, 3, 11, 12])]
    cooling = act[act.month.isin([6, 7, 8])]
    return {
        "always_bid": act["P"].mean(),
        "day": day["P"].mean(),
        "night": night["P"].mean(),
        "heating_season": heating["P"].mean(),
        "cooling_season": cooling["P"].mean(),
        "selective_top25": act[act.P >= act.P.quantile(0.75)]["P"].mean(),
        "selective_top10": act[act.P >= act.P.quantile(0.90)]["P"].mean(),
        "activation_freq": len(act) / len(df),
    }


def device_value(kwh, price_eur_mwh):
    """annual_value[EUR] = kWh/yr * price[EUR/MWh] / 1000."""
    return kwh * price_eur_mwh / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Ausgleichsenergie-und-Regelenergie-2026.csv")
    args = ap.parse_args()

    df = load_prices(args.csv)
    cp = capture_prices(df)

    span_days = (df.dt.max() - df.dt.min()).days + 1
    print(f"Data period: {df.dt.min().date()} -> {df.dt.max().date()} "
          f"({span_days} days, 15-min resolution)")
    print(f"mFRR+ activation frequency: {cp['activation_freq']:.1%} of intervals\n")

    print("CAPTURE PRICES (EUR/MWh, energy paid only when activated)")
    for k in ["always_bid", "day", "night", "heating_season",
              "cooling_season", "selective_top25", "selective_top10"]:
        print(f"  {k:18s}: {cp[k]:7.1f}")
    print()

    # Choose a sensible capture price per device:
    #   - HP uses heating-season price; AC uses cooling-season price;
    #   - EV/battery use the all-year always-bid price.
    season_price = {
        "heating": cp["heating_season"],
        "cooling": cp["cooling_season"],
        "all": cp["always_bid"],
    }

    print(f"{'Device':30s} {'Profile':9s} {'kWh/yr':>8s} "
          f"{'EUR/MWh':>9s} {'EUR/yr':>9s}")
    print("-" * 70)
    totals = {"Standard": 0.0, "Tolerant": 0.0}
    for name, kw, std_kwh, tol_kwh, season in DEVICES:
        price = season_price[season]
        for label, kwh in [("Standard", std_kwh), ("Tolerant", tol_kwh)]:
            val = device_value(kwh, price)
            totals[label] += val
            print(f"{name:30s} {label:9s} {kwh:8.0f} {price:9.1f} {val:9.1f}")
    print("-" * 70)
    print(f"{'TOTAL per household':30s} {'Standard':9s} {'':8s} {'':9s} "
          f"{totals['Standard']:9.1f}")
    print(f"{'TOTAL per household':30s} {'Tolerant':9s} {'':8s} {'':9s} "
          f"{totals['Tolerant']:9.1f}")
    print("\nValues in EUR/yr. Energy-only mFRR+ revenue; excludes capacity "
          "(reserve-holding) payments, aggregator fees and activation losses.")


if __name__ == "__main__":
    main()