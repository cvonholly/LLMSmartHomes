"""
make_figures.py — Publication-quality figures for the Moneta / PLEMSli
Home Flexibility Economics Model. All numbers sourced from
PLEMSli_Economic_Model_and_Business_Case.xlsx.

Outputs PDF (vector, for LaTeX \includegraphics) + PNG (preview) per figure.
Terminology matches the report: Consumer / Aggregator products,
Conservative / Tolerant occupant profiles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

OUT = Path("figures"); OUT.mkdir(exist_ok=True)

# ---- Global style: clean, LaTeX-friendly, colour-blind-safe palette ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 120,
})

# Palette
C_CONS   = "#2C6E8F"   # Consumer product / conservative — teal-blue
C_AGG    = "#C25B3F"   # Aggregator product — warm terracotta
C_STD    = "#5B8DB8"   # standard/conservative profile
C_TOL    = "#E0A458"   # tolerant profile
GREY     = "#6B6B6B"
POS      = "#3C7A5E"   # positive profit
NEG      = "#B0483A"   # negative / loss

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)

def chf(x, _=None):
    if abs(x) >= 1e6: return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3: return f"{x/1e3:.0f}k"
    return f"{x:.0f}"

# ======================================================================
# DATA (from spreadsheet)
# ======================================================================
devices = ["Heat pump", "EV", "Battery", "A/C"]

# Product 1 (Consumer) per-home value CHF/yr, by profile  [Sheet 3]
# CKW data:
# p1_std = [59.6, 147.66, 110.187, 11.92]
# p1_tol = [119.2, 232.3, 110.187, 23.84]
# EKZ data:
p1_std = [44, 173, 0, 9]
p1_tol = [87, 182, 0, 17]
# Product 2 (Aggregator) per-home value CHF/yr, by profile [Sheet 4]
p2_std = [51.3728, 295.3936, 49.10004, 10.27456]
p2_tol = [102.7456, 295.3936, 49.10004, 20.54912]

# CO2 savings kg/home/yr, by profile (same for both products) [Sheet 3/4]
co2_std = [16.624, 95.588, 0.0, 3.3248]
co2_tol = [33.248, 95.588, 0.0, 6.6496]

# Business case 5-year ramp [Sheet 6]
years = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
active_homes = [12.35, 130.815, 416.9835, 1572.28515, 6203.056635]

p1_rev  = [2772.502841, 29367.20317, 93610.35937, 352968.8295, 1392549.971]
p1_cost = [101254.5, 112937.05, 138638.845, 247859.9605, 685413.9645]
p1_gp   = [-98481.99716, -83569.84683, -45028.48563, 105108.869, 707136.0065]
p1_cum  = [-98481.99716, -182051.844, -227080.3296, -121971.4606, 585164.546]

p2_rev  = [3155.664731, 33425.77181, 106547.3785, 401749.3761, 1585001.38]
p2_cost = [101254.5, 112937.05, 138638.845, 247859.9605, 685413.9645]
p2_gp   = [-98098.83527, -79511.27819, -32091.4665, 153889.4156, 899587.416]
p2_cum  = [-98098.83527, -177610.1135, -209701.58, -55812.16434, 843775.2516]

# Blended per-home revenue [Sheet 6]
p1_per_home = 224.4941571
p2_per_home = 255.5194114

# Scenario: consumer tariff geography (EKZ Zurich vs CKW Luzern) [README/report]
# Year-5 GP: EKZ ~707k -> CKW scaled by spread factor 1.366972477
scn_gp = {"Consumer\n(EKZ / Zurich)": 275300,
          "Consumer\n(CKW / Luzern)": 707100,
          "Aggregator\n(mFRR)": 899587.416}

# Market reach [Sheet 5]
mr_dev = ["Heat pump", "EV", "Home battery", "A/C", "PV"]
mr_installed   = [280000, 145000, 160000, 50000, 222000]
mr_serviceable = [126000, 65250, 72000, 22500, 99900]
mr_target      = [6300, 3262.5, 3600, 1125, 4995]

# ======================================================================
# FIG 1 — Per-home value by device & product (grouped bars, 2 panels)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
x = np.arange(len(devices)); w = 0.38

for ax, std, tol, title, base in [
    (axes[0], p1_std, p1_tol, "Consumer product", C_CONS),
    (axes[1], p2_std, p2_tol, "Aggregator product", C_AGG)]:
    b1 = ax.bar(x - w/2, std, w, label="Conservative", color=C_STD, edgecolor="white")
    b2 = ax.bar(x + w/2, tol, w, label="Tolerant", color=C_TOL, edgecolor="white")
    ax.set_title(title, color=base)
    ax.set_xticks(x); ax.set_xticklabels(devices, rotation=15, ha="right")
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.0f}", (r.get_x()+r.get_width()/2, h),
                        textcoords="offset points", xytext=(0,2),
                        ha="center", fontsize=8, color=GREY)
    ax.legend(loc="upper left")

axes[0].set_ylabel("Value per home (CHF / yr)")
fig.suptitle("Per-home annual value by device and occupant profile",
             fontsize=13, fontweight="bold", y=1.02)
save(fig, "fig1_device_value_by_product")

# ======================================================================
# FIG 2 — Total per-home revenue: Consumer vs Aggregator × profile (stacked)
# ======================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.4))
labels = ["Consumer (EKZ)\nConservative", "Consumer (EKZ)\nTolerant",
          "Aggregator\nConservative", "Aggregator\nTolerant"]
data = [p1_std, p1_tol, p2_std, p2_tol]
dev_colors = ["#2C6E8F", "#4E9DB8", "#8FBFD0", "#CFE3EA"]
bottoms = np.zeros(4)
for di, dev in enumerate(devices):
    vals = [d[di] for d in data]
    ax.bar(labels, vals, bottom=bottoms, label=dev,
           color=dev_colors[di], edgecolor="white", linewidth=0.6)
    bottoms += np.array(vals)
for i, tot in enumerate(bottoms):
    ax.annotate(f"{tot:.0f}", (i, tot), textcoords="offset points",
                xytext=(0,3), ha="center", fontweight="bold", fontsize=9)
ax.set_ylabel("Total value per home (CHF / yr)")
ax.set_title("Total per-home value by product and profile, split by device")
ax.legend(title="Device", loc="upper left", ncol=2, fontsize=9)
save(fig, "fig2_total_value_stacked")

# ======================================================================
# FIG 3 — 5-year business case: cumulative gross profit (break-even)
# ======================================================================
fig, ax = plt.subplots(figsize=(7.6, 4.4))
xs = np.arange(len(years))
ax.plot(xs, p1_cum, "-o", color=C_CONS, lw=2.2, label="Consumer (Product 1)")
ax.plot(xs, p2_cum, "-s", color=C_AGG, lw=2.2, label="Aggregator (Product 2)")
ax.axhline(0, color=GREY, lw=1, ls="--")
ax.fill_between(xs, p1_cum, 0, where=np.array(p1_cum)<0, color=C_CONS, alpha=0.07)
# annotate year-5 endpoints
for cum, col, dy in [(p1_cum, C_CONS, 12), (p2_cum, C_AGG, -18)]:
    ax.annotate(f"{cum[-1]/1e3:.0f}k CHF", (4, cum[-1]),
                textcoords="offset points", xytext=(-4, dy),
                ha="right", color=col, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(years)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: chf(v)))
ax.set_ylabel("Cumulative gross profit (CHF)")
ax.set_title("Cumulative gross profit over the 5-year ramp (CH)")
ax.legend(loc="upper left")
ax.text(0.5, -227080, "trough:\ncash low-point", fontsize=8, color=GREY, ha="center")
save(fig, "fig3_cumulative_gross_profit")

# ======================================================================
# FIG 4 — Year-5 revenue / gross profit comparison (grouped)
# ======================================================================
fig, ax = plt.subplots(figsize=(6.8, 4.3))
groups = ["Year-5\nrevenue", "Year-5\ngross profit", "Cumulative 5-yr\ngross profit"]
p1v = [1392549.971, 707136.0065, 585164.546]
p2v = [1585001.38, 899587.416, 843775.2516]
x = np.arange(len(groups)); w = 0.38
b1 = ax.bar(x-w/2, p1v, w, label="Consumer (Product 1)", color=C_CONS, edgecolor="white")
b2 = ax.bar(x+w/2, p2v, w, label="Aggregator (Product 2)", color=C_AGG, edgecolor="white")
for bars in (b1,b2):
    for r in bars:
        h=r.get_height()
        ax.annotate(chf(h), (r.get_x()+r.get_width()/2, h),
                    textcoords="offset points", xytext=(0,3),
                    ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: chf(v)))
ax.set_ylabel("CHF")
ax.set_title("Product comparison at Year 5")
ax.legend(loc="upper left")
save(fig, "fig4_year5_comparison")

# ======================================================================
# FIG 5 — Scenario sensitivity: Year-5 gross profit (tariff geography)
# ======================================================================
fig, ax = plt.subplots(figsize=(6.8, 4.0))
names = list(scn_gp.keys()); vals = list(scn_gp.values())
cols = [C_CONS, "#7FB0C6", C_AGG]
bars = ax.barh(names, vals, color=cols, edgecolor="white")
for r in bars:
    wv=r.get_width()
    ax.annotate(f"{wv/1e3:.0f}k", (wv, r.get_y()+r.get_height()/2),
                textcoords="offset points", xytext=(4,0), va="center",
                fontweight="bold", fontsize=9)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: chf(v)))
ax.set_xlabel("Year-5 gross profit (CHF)")
ax.set_title("Scenario sensitivity: tariff geography vs. market channel")
ax.invert_yaxis()
ax.margins(x=0.15)
save(fig, "fig5_scenario_sensitivity")

# ======================================================================
# FIG 6 — Market reach funnel (installed -> serviceable -> target)
# ======================================================================
fig, ax = plt.subplots(figsize=(7.8, 4.4))
x = np.arange(len(mr_dev)); w = 0.26
ax.bar(x-w, mr_installed, w, label="Installed base (2026)", color="#B9CBD6", edgecolor="white")
ax.bar(x,   mr_serviceable, w, label="Serviceable (smart meter + dyn. tariff)", color=C_CONS, edgecolor="white")
ax.bar(x+w, mr_target, w, label="Moneta target (5% of serviceable)", color=C_AGG, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(mr_dev)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: chf(v)))
ax.set_ylabel("Swiss single-family homes")
ax.set_title("Market reach funnel by device segment (Switzerland)")
ax.legend(loc="upper right", fontsize=9)
save(fig, "fig6_market_reach_funnel")

# ======================================================================
# FIG 7 — CO2 savings per home by device & profile
# ======================================================================
fig, ax = plt.subplots(figsize=(7.0, 4.2))
x = np.arange(len(devices)); w=0.38
b1=ax.bar(x-w/2, co2_std, w, label="Conservative", color=C_STD, edgecolor="white")
b2=ax.bar(x+w/2, co2_tol, w, label="Tolerant", color=C_TOL, edgecolor="white")
for bars in (b1,b2):
    for r in bars:
        h=r.get_height()
        if h>0:
            ax.annotate(f"{h:.0f}", (r.get_x()+r.get_width()/2,h),
                        textcoords="offset points", xytext=(0,2),
                        ha="center", fontsize=8, color=GREY)
tot_std=sum(co2_std); tot_tol=sum(co2_tol)
ax.set_xticks(x); ax.set_xticklabels(devices)
ax.set_ylabel("CO$_2$ savings (kg / home / yr)")
ax.set_title(f"CO$_2$ savings per home by device\n(total: {tot_std:.0f} kg conservative, {tot_tol:.0f} kg tolerant)")
ax.legend(loc="upper right")
save(fig, "fig7_co2_savings")

print("\nAll figures written to", OUT.resolve())
