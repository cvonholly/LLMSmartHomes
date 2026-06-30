"""
fetch_ekz_dynamic_2026.py — Pull EKZ dynamic electricity tariff prices for all of 2026.

Uses the public /v1/tariffs endpoint with tariff_name=electricity_dynamic.
Tariffs are published in 15-min resolution (96/day); this script aggregates
to hourly averages.

Note: dynamic tariffs are only published daily until 18:00 for the next day,
so data for future dates beyond what EKZ has published will be empty.
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BASE_URL = "https://api.tariffs.ekz.ch/v1/tariffs"
    # TARIFF_NAME = "electricity_dynamic"
TARIFF_NAME = "integrated_400D"
TARIFF_TYPE = "integrated"  # full total price (grid + electricity)


def fmt_ts(dt: datetime) -> str:
    # ISO 8601 with offset, e.g. 2026-01-01T00:00:00+01:00
    return dt.strftime("%Y-%m-%dT%H:%M:%S%z")


def fetch_day(session: requests.Session, day: datetime, offset_hours: int):
    """Fetch all prices for a single calendar day (local Swiss time)."""
    tz = timezone(timedelta(hours=offset_hours))
    start = day.replace(hour=0, minute=0, second=0, tzinfo=tz)
    end = day.replace(hour=23, minute=59, second=59, tzinfo=tz)

    params = {
        "tariff_type": TARIFF_TYPE,
        "tariff_name": TARIFF_NAME,
        "start_timestamp": fmt_ts(start),
        "end_timestamp": fmt_ts(end),
    }
    r = session.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("prices", [])


def swiss_offset(day: datetime) -> int:
    """Return UTC offset in hours for CET/CEST on a given day (approx DST rule)."""
    # DST: last Sunday March -> last Sunday October
    year = day.year
    # last Sunday of March
    mar = datetime(year, 3, 31)
    mar -= timedelta(days=(mar.weekday() + 1) % 7)
    # last Sunday of October
    oct_ = datetime(year, 10, 31)
    oct_ -= timedelta(days=(oct_.weekday() + 1) % 7)
    return 2 if mar.date() <= day.date() < oct_.date() else 1


def fetch_range(session, start, end):
    """Fetch all prices in one request for an arbitrary timestamp range."""
    params = {
        "tariff_type": TARIFF_TYPE,
        "tariff_name": TARIFF_NAME,
        "start_timestamp": fmt_ts(start),
        "end_timestamp": fmt_ts(end),
    }
    r = session.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("prices", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--out", default="ekz_dynamic_2026_hourly.csv")
    ap.add_argument("--raw-out", default="ekz_dynamic_2026_15min.csv")
    ap.add_argument("--chunk", choices=["year", "month"], default="month",
                    help="request the whole year at once, or month by month")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    rows = []
    if args.chunk == "year":
        spans = [(datetime(args.year, 1, 1), datetime(args.year, 12, 31))]
    else:
        spans = []
        for m in range(1, 13):
            start = datetime(args.year, m, 1)
            end = (datetime(args.year, m + 1, 1) if m < 12
                   else datetime(args.year + 1, 1, 1)) - timedelta(days=1)
            spans.append((start, end))

    for start_d, end_d in spans:
        off_start = swiss_offset(start_d)
        off_end = swiss_offset(end_d)
        start = start_d.replace(hour=0, minute=0, second=0,
                                tzinfo=timezone(timedelta(hours=off_start)))
        end = end_d.replace(hour=23, minute=59, second=59,
                            tzinfo=timezone(timedelta(hours=off_end)))
        try:
            prices = fetch_range(session, start, end)
        except requests.HTTPError as e:
            print(f"[{start_d:%Y-%m} .. {end_d:%Y-%m-%d}] HTTP error: {e}")
            continue

        for p in prices:
            integ = p.get("integrated") or []
            value = next((e["value"] for e in integ if e["unit"] == "CHF_kWh"), None)
            rows.append({
                "start_timestamp": p["start_timestamp"],
                "end_timestamp": p["end_timestamp"],
                "price_chf_kwh": value,
            })
        print(f"[{start_d:%Y-%m-%d} .. {end_d:%Y-%m-%d}] {len(prices)} intervals")

    if not rows:
        print("No data returned.")
        return

    df = pd.DataFrame(rows)
    # utc=True resolves the mixed +01:00 / +02:00 offsets, then back to Swiss local
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], utc=True)
    df["end_timestamp"] = pd.to_datetime(df["end_timestamp"], utc=True)
    df["start_timestamp"] = df["start_timestamp"].dt.tz_convert("Europe/Zurich")
    df["end_timestamp"] = df["end_timestamp"].dt.tz_convert("Europe/Zurich")
    df = df.drop_duplicates("start_timestamp").sort_values("start_timestamp").reset_index(drop=True)
    df.to_csv(args.raw_out, index=False)
    print(f"\nWrote {len(df)} 15-min intervals -> {args.raw_out}")

    hourly = (
        df.set_index("start_timestamp")["price_chf_kwh"]
        .resample("1h")
        .mean()
        .reset_index()
        .rename(columns={"start_timestamp": "hour", "price_chf_kwh": "avg_price_chf_kwh"})
    )
    hourly.to_csv(args.out, index=False)
    print(f"Wrote {len(hourly)} hourly rows -> {args.out}")


if __name__ == "__main__":
    main()