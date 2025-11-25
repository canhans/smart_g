# pdlc_app_scenarios.py
# Streamlit web app – Compare two scenarios:
# Scenario A: "Standard office"
# Scenario B: "Premium HQ"
#
# PDLC vs mechanical/roller blinds value proposition

from dataclasses import dataclass
from typing import Dict, Tuple
import streamlit as st
import pandas as pd

# -----------------------------
# 1) Data models
# -----------------------------

@dataclass
class ProjectInputs:
    n_rooms: int
    area_glass_per_room_m2: float
    time_horizon_years: int
    discount_rate: float          # e.g. 0.08 for 8%
    labour_cost_per_hour: float
    electricity_price_per_kwh: float


@dataclass
class BlindsInputs:
    capex_blinds_per_m2: float
    install_cost_per_room: float

    lifetime_years: int
    failures_per_room_per_year: float
    repair_cost_per_failure: float

    cleaning_hours_per_room_per_year: float
    cleaning_material_cost_per_room_per_year: float

    downtime_hours_per_failure: float
    cost_per_downtime_hour: float


@dataclass
class PdlcInputs:
    capex_pdlc_per_m2: float
    install_cost_per_room: float

    # Energy
    pdlc_power_w_per_m2: float
    pdlc_on_hours_per_day: float

    # Maintenance / drivers
    driver_cost_ratio: float            # share of driver in total PDLC capex (e.g. 0.15)
    driver_lifetime_years: int
    annual_maintenance_cost_per_room: float  # other small checks, etc.


@dataclass
class QualitativeScorecard:
    # weights can be arbitrary; we normalize them inside the function
    weights: Dict[str, float]
    scores_blinds: Dict[str, float]    # 1–5
    scores_pdlc: Dict[str, float]      # 1–5


# -----------------------------
# 2) Helper & core calculations
# -----------------------------

def npv_of_constant_cashflow(annual_cashflow: float, years: int, discount_rate: float) -> float:
    """NPV of a constant annual cash flow over 'years' with discount rate r."""
    r = discount_rate
    if r == 0:
        return annual_cashflow * years
    factor = (1 - (1 + r) ** (-years)) / r
    return annual_cashflow * factor


def capex_blinds(project: ProjectInputs, blinds: BlindsInputs) -> float:
    return project.n_rooms * (
        project.area_glass_per_room_m2 * blinds.capex_blinds_per_m2
        + blinds.install_cost_per_room
    )


def capex_pdlc(project: ProjectInputs, pdlc: PdlcInputs) -> float:
    return project.n_rooms * (
        project.area_glass_per_room_m2 * pdlc.capex_pdlc_per_m2
        + pdlc.install_cost_per_room
    )


def annual_opex_blinds(
    project: ProjectInputs,
    blinds: BlindsInputs,
    capex_blinds_total: float
) -> Tuple[float, Dict[str, float]]:
    """Returns (total_annual_opex, breakdown_dict)"""

    # Cleaning cost
    cleaning_cost = project.n_rooms * (
        blinds.cleaning_hours_per_room_per_year * project.labour_cost_per_hour
        + blinds.cleaning_material_cost_per_room_per_year
    )

    # Repairs
    repair_cost = project.n_rooms * (
        blinds.failures_per_room_per_year * blinds.repair_cost_per_failure
    )

    # Replacement (straight-line over lifetime)
    replacement_cost = capex_blinds_total / blinds.lifetime_years

    # Downtime
    downtime_cost = project.n_rooms * (
        blinds.failures_per_room_per_year
        * blinds.downtime_hours_per_failure
        * blinds.cost_per_downtime_hour
    )

    total = cleaning_cost + repair_cost + replacement_cost + downtime_cost
    breakdown = {
        "Cleaning": cleaning_cost,
        "Repairs": repair_cost,
        "Replacement": replacement_cost,
        "Downtime": downtime_cost,
    }
    return total, breakdown


def annual_opex_pdlc(
    project: ProjectInputs,
    pdlc: PdlcInputs,
    capex_pdlc_total: float
) -> Tuple[float, Dict[str, float]]:
    """Returns (total_annual_opex, breakdown_dict)"""

    # Energy consumption
    energy_kwh_per_year = (
        project.n_rooms
        * project.area_glass_per_room_m2
        * pdlc.pdlc_power_w_per_m2
        * pdlc.pdlc_on_hours_per_day
        * 365
        / 1000.0
    )
    energy_cost = energy_kwh_per_year * project.electricity_price_per_kwh

    # Driver replacement (straight-line on driver share of capex)
    driver_capex = capex_pdlc_total * pdlc.driver_cost_ratio
    driver_replacement_cost = driver_capex / pdlc.driver_lifetime_years

    # Other maintenance
    maintenance_cost = project.n_rooms * pdlc.annual_maintenance_cost_per_room

    total = energy_cost + driver_replacement_cost + maintenance_cost
    breakdown = {
        "Energy": energy_cost,
        "Driver replacement": driver_replacement_cost,
        "Maintenance": maintenance_cost,
    }
    return total, breakdown


def compute_tco_and_metrics(
    project: ProjectInputs,
    blinds: BlindsInputs,
    pdlc: PdlcInputs,
):
    """High-level wrapper: returns TCOs, savings, ROI, payback, + Opex breakdowns."""
    # Capex
    capex_blinds_total = capex_blinds(project, blinds)
    capex_pdlc_total = capex_pdlc(project, pdlc)

    # Annual Opex
    opex_blinds, opex_blinds_breakdown = annual_opex_blinds(project, blinds, capex_blinds_total)
    opex_pdlc, opex_pdlc_breakdown = annual_opex_pdlc(project, pdlc, capex_pdlc_total)

    # NPV Opex
    npv_opex_blinds = npv_of_constant_cashflow(
        opex_blinds, project.time_horizon_years, project.discount_rate
    )
    npv_opex_pdlc = npv_of_constant_cashflow(
        opex_pdlc, project.time_horizon_years, project.discount_rate
    )

    # TCO
    tco_blinds = capex_blinds_total + npv_opex_blinds
    tco_pdlc = capex_pdlc_total + npv_opex_pdlc

    # Savings & ROI
    npv_savings_vs_blinds = tco_blinds - tco_pdlc
    incremental_capex = capex_pdlc_total - capex_blinds_total

    if incremental_capex != 0:
        roi = npv_savings_vs_blinds / incremental_capex
    else:
        roi = float("inf") if npv_savings_vs_blinds > 0 else 0.0

    # Simple (non-discounted) payback
    annual_savings = opex_blinds - opex_pdlc
    if annual_savings > 0:
        payback_years = incremental_capex / annual_savings
    else:
        payback_years = float("inf")

    return {
        "capex_blinds_total": capex_blinds_total,
        "capex_pdlc_total": capex_pdlc_total,
        "opex_blinds_annual": opex_blinds,
        "opex_pdlc_annual": opex_pdlc,
        "opex_blinds_breakdown": opex_blinds_breakdown,
        "opex_pdlc_breakdown": opex_pdlc_breakdown,
        "tco_blinds": tco_blinds,
        "tco_pdlc": tco_pdlc,
        "npv_savings_vs_blinds": npv_savings_vs_blinds,
        "incremental_capex": incremental_capex,
        "roi": roi,
        "payback_years": payback_years,
    }


def compute_qualitative_scores(scorecard: QualitativeScorecard):
    """
    Returns (score_blinds, score_pdlc, normalized_weights_dict).
    We normalize weights so their sum = 1.
    """
    total_w = sum(scorecard.weights.values())
    if total_w <= 0:
        norm_weights = {k: 0.0 for k in scorecard.weights.keys()}
    else:
        norm_weights = {k: w / total_w for k, w in scorecard.weights.items()}

    score_blinds = 0.0
    score_pdlc = 0.0
    for crit, w_norm in norm_weights.items():
        s_b = scorecard.scores_blinds.get(crit, 0.0)
        s_p = scorecard.scores_pdlc.get(crit, 0.0)
        score_blinds += w_norm * s_b
        score_pdlc += w_norm * s_p

    return score_blinds, score_pdlc, norm_weights


# -----------------------------
# 3) Streamlit UI
# -----------------------------

def run_app():
    st.set_page_config(page_title="PDLC vs Blinds – Scenario Comparison", layout="wide")

    st.title("PDLC vs Mechanical/Roller Blinds – Scenario Comparison")

    st.markdown(
        "Compare PDLC vs blinds in two different project types:\n\n"
        "- **Scenario A: Standard office**\n"
        "- **Scenario B: Premium HQ**\n\n"
        "Global assumptions are shared. You can tune size and capex levels per scenario."
    )

    # ------------ Sidebar: shared + scenario-specific inputs ------------
    with st.sidebar:
        st.header("Global assumptions (shared)")

        horizon = st.number_input("Time horizon (years)", min_value=1, value=10)
        discount_rate = (
            st.number_input("Discount rate (%)", min_value=0.0, max_value=50.0, value=8.0) / 100.0
        )
        labour_cost = st.number_input("Labour cost (€/hour)", min_value=0.0, value=20.0)
        electricity_price = st.number_input(
            "Electricity price (€/kWh)", min_value=0.0, value=0.20
        )

        st.subheader("Blinds assumptions (shared)")
        lifetime_blinds = st.number_input("Blinds lifetime (years)", min_value=1, value=7)
        failures_per_room = st.number_input(
            "Failures per room per year", min_value=0.0, value=0.3
        )
        repair_cost = st.number_input(
            "Repair cost per failure (€)", min_value=0.0, value=80.0
        )
        cleaning_hours = st.number_input(
            "Cleaning hours per room per year", min_value=0.0, value=1.0
        )
        cleaning_material = st.number_input(
            "Cleaning material cost per room per year (€)", min_value=0.0, value=10.0
        )
        downtime_hours = st.number_input(
            "Downtime hours per failure", min_value=0.0, value=1.0
        )
        downtime_cost = st.number_input(
            "Cost per downtime hour (€)", min_value=0.0, value=200.0
        )

        st.subheader("PDLC assumptions (shared)")
        pdlc_power = st.number_input("PDLC power (W/m²)", min_value=0.0, value=5.0)
        pdlc_hours = st.number_input("PDLC ON hours per day", min_value=0.0, value=4.0)
        driver_ratio = st.number_input(
            "Driver cost ratio of PDLC capex", min_value=0.0, max_value=1.0, value=0.15
        )
        driver_life = st.number_input("Driver lifetime (years)", min_value=1, value=10)
        maintenance_pdlc = st.number_input(
            "Other PDLC maintenance per room per year (€)", min_value=0.0, value=5.0
        )

        st.markdown("---")
        st.header("Scenario A – Standard office")

        n_rooms_A = st.number_input("Rooms (Standard office)", min_value=1, value=10)
        area_A = st.number_input("Glass area per room A (m²)", min_value=1.0, value=10.0)
        capex_blinds_A_m2 = st.number_input(
            "Blinds capex A (€/m²)", min_value=0.0, value=80.0
        )
        install_blinds_A = st.number_input(
            "Install cost per room A – blinds (€)", min_value=0.0, value=150.0
        )
        capex_pdlc_A_m2 = st.number_input(
            "PDLC capex A (€/m²)", min_value=0.0, value=300.0
        )
        install_pdlc_A = st.number_input(
            "Install cost per room A – PDLC (€)", min_value=0.0, value=250.0
        )

        st.markdown("---")
        st.header("Scenario B – Premium HQ")

        n_rooms_B = st.number_input("Rooms (Premium HQ)", min_value=1, value=20)
        area_B = st.number_input("Glass area per room B (m²)", min_value=1.0, value=15.0)
        capex_blinds_B_m2 = st.number_input(
            "Blinds capex B (€/m²)", min_value=0.0, value=120.0
        )
        install_blinds_B = st.number_input(
            "Install cost per room B – blinds (€)", min_value=0.0, value=250.0
        )
        capex_pdlc_B_m2 = st.number_input(
            "PDLC capex B (€/m²)", min_value=0.0, value=400.0
        )
        install_pdlc_B = st.number_input(
            "Install cost per room B – PDLC (€)", min_value=0.0, value=350.0
        )

    # ------------ Build shared base assumptions ------------
    # (We will plug in scenario-specific rooms/areas/capex)

    blinds_base = BlindsInputs(
        capex_blinds_per_m2=0.0,          # overwrite below per scenario
        install_cost_per_room=0.0,        # overwrite below per scenario
        lifetime_years=lifetime_blinds,
        failures_per_room_per_year=failures_per_room,
        repair_cost_per_failure=repair_cost,
        cleaning_hours_per_room_per_year=cleaning_hours,
        cleaning_material_cost_per_room_per_year=cleaning_material,
        downtime_hours_per_failure=downtime_hours,
        cost_per_downtime_hour=downtime_cost,
    )

    pdlc_base = PdlcInputs(
        capex_pdlc_per_m2=0.0,            # overwrite below per scenario
        install_cost_per_room=0.0,        # overwrite below per scenario
        pdlc_power_w_per_m2=pdlc_power,
        pdlc_on_hours_per_day=pdlc_hours,
        driver_cost_ratio=driver_ratio,
        driver_lifetime_years=driver_life,
        annual_maintenance_cost_per_room=maintenance_pdlc,
    )

    # ------------ Scenario A – Standard office ------------
    project_A = ProjectInputs(
        n_rooms=n_rooms_A,
        area_glass_per_room_m2=area_A,
        time_horizon_years=horizon,
        discount_rate=discount_rate,
        labour_cost_per_hour=labour_cost,
        electricity_price_per_kwh=electricity_price,
    )

    blinds_A = BlindsInputs(
        capex_blinds_per_m2=capex_blinds_A_m2,
        install_cost_per_room=install_blinds_A,
        lifetime_years=blinds_base.lifetime_years,
        failures_per_room_per_year=blinds_base.failures_per_room_per_year,
        repair_cost_per_failure=blinds_base.repair_cost_per_failure,
        cleaning_hours_per_room_per_year=blinds_base.cleaning_hours_per_room_per_year,
        cleaning_material_cost_per_room_per_year=blinds_base.cleaning_material_cost_per_room_per_year,
        downtime_hours_per_failure=blinds_base.downtime_hours_per_failure,
        cost_per_downtime_hour=blinds_base.cost_per_downtime_hour,
    )

    pdlc_A = PdlcInputs(
        capex_pdlc_per_m2=capex_pdlc_A_m2,
        install_cost_per_room=install_pdlc_A,
        pdlc_power_w_per_m2=pdlc_base.pdlc_power_w_per_m2,
        pdlc_on_hours_per_day=pdlc_base.pdlc_on_hours_per_day,
        driver_cost_ratio=pdlc_base.driver_cost_ratio,
        driver_lifetime_years=pdlc_base.driver_lifetime_years,
        annual_maintenance_cost_per_room=pdlc_base.annual_maintenance_cost_per_room,
    )

    results_A = compute_tco_and_metrics(project_A, blinds_A, pdlc_A)

    # ------------ Scenario B – Premium HQ ------------
    project_B = ProjectInputs(
        n_rooms=n_rooms_B,
        area_glass_per_room_m2=area_B,
        time_horizon_years=horizon,
        discount_rate=discount_rate,
        labour_cost_per_hour=labour_cost,
        electricity_price_per_kwh=electricity_price,
    )

    blinds_B = BlindsInputs(
        capex_blinds_per_m2=capex_blinds_B_m2,
        install_cost_per_room=install_blinds_B,
        lifetime_years=blinds_base.lifetime_years,
        failures_per_room_per_year=blinds_base.failures_per_room_per_year,
        repair_cost_per_failure=blinds_base.repair_cost_per_failure,
        cleaning_hours_per_room_per_year=blinds_base.cleaning_hours_per_room_per_year,
        cleaning_material_cost_per_room_per_year=blinds_base.cleaning_material_cost_per_room_per_year,
        downtime_hours_per_failure=blinds_base.downtime_hours_per_failure,
        cost_per_downtime_hour=blinds_base.cost_per_downtime_hour,
    )

    pdlc_B = PdlcInputs(
        capex_pdlc_per_m2=capex_pdlc_B_m2,
        install_cost_per_room=install_pdlc_B,
        pdlc_power_w_per_m2=pdlc_base.pdlc_power_w_per_m2,
        pdlc_on_hours_per_day=pdlc_base.pdlc_on_hours_per_day,
        driver_cost_ratio=pdlc_base.driver_cost_ratio,
        driver_lifetime_years=pdlc_base.driver_lifetime_years,
        annual_maintenance_cost_per_room=pdlc_base.annual_maintenance_cost_per_room,
    )

    results_B = compute_tco_and_metrics(project_B, blinds_B, pdlc_B)

    # ------------ Main layout: two columns (A vs B) ------------
    st.markdown("## Financial comparison – Scenario A vs Scenario B")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Scenario A – Standard office")
        st.metric("TCO – Blinds", f"{results_A['tco_blinds']:,.0f} €")
        st.metric("TCO – PDLC", f"{results_A['tco_pdlc']:,.0f} €")
        st.metric("NPV savings vs blinds", f"{results_A['npv_savings_vs_blinds']:,.0f} €")
        if results_A["payback_years"] != float("inf"):
            st.metric("Simple payback (years)", f"{results_A['payback_years']:.1f}")
        else:
            st.metric("Simple payback (years)", "No payback")

        st.markdown("**Annual Opex breakdown (A)**")
        opex_A_df = pd.DataFrame({
            "Blinds A": results_A["opex_blinds_breakdown"],
            "PDLC A": results_A["opex_pdlc_breakdown"],
        })
        st.bar_chart(opex_A_df)

    with colB:
        st.markdown("### Scenario B – Premium HQ")
        st.metric("TCO – Blinds", f"{results_B['tco_blinds']:,.0f} €")
        st.metric("TCO – PDLC", f"{results_B['tco_pdlc']:,.0f} €")
        st.metric("NPV savings vs blinds", f"{results_B['npv_savings_vs_blinds']:,.0f} €")
        if results_B["payback_years"] != float("inf"):
            st.metric("Simple payback (years)", f"{results_B['payback_years']:.1f}")
        else:
            st.metric("Simple payback (years)", "No payback")

        st.markdown("**Annual Opex breakdown (B)**")
        opex_B_df = pd.DataFrame({
            "Blinds B": results_B["opex_blinds_breakdown"],
            "PDLC B": results_B["opex_pdlc_breakdown"],
        })
        st.bar_chart(opex_B_df)

    # ------------ Qualitative scorecard (same PDLC vs blinds tech for both scenarios) ------------
    st.markdown("---")
    st.subheader("Qualitative value of PDLC vs blinds (same for both scenarios)")

    with st.expander("Adjust qualitative scores"):
        default_weights = {
            "Privacy": 0.25,
            "Aesthetics": 0.25,
            "Hygiene": 0.20,
            "Flexibility": 0.20,
            "Innovation": 0.10,
        }
        weights: Dict[str, float] = {}
        scores_blinds: Dict[str, float] = {}
        scores_pdlc: Dict[str, float] = {}

        st.write("**Weights** (sliders; app will normalize them internally)**")
        for crit, w in default_weights.items():
            weights[crit] = st.slider(
                f"Weight – {crit}", min_value=0.0, max_value=1.0, value=float(w), step=0.05
            )

        st.write("**Scores (1–5)**")
        cols = st.columns(3)
        for idx, crit in enumerate(default_weights.keys()):
            with cols[idx % 3]:
                scores_blinds[crit] = st.slider(
                    f"{crit} – Blinds", min_value=1.0, max_value=5.0, value=3.0, step=0.5
                )
                scores_pdlc[crit] = st.slider(
                    f"{crit} – PDLC", min_value=1.0, max_value=5.0, value=5.0, step=0.5
                )

        scorecard = QualitativeScorecard(
            weights=weights,
            scores_blinds=scores_blinds,
            scores_pdlc=scores_pdlc,
        )

        score_blinds, score_pdlc, norm_weights = compute_qualitative_scores(scorecard)

        st.write("**Normalized weights actually used (sum = 1):**")
        st.json(norm_weights)

        st.write(f"**Qualitative score – Blinds:** {score_blinds:.2f}")
        st.write(f"**Qualitative score – PDLC:** {score_pdlc:.2f}")

        # Progress bars relative to the higher score so changes are visible
        max_score = max(score_blinds, score_pdlc, 1.0)
        st.progress(score_blinds / max_score, text="Blinds (relative)")
        st.progress(score_pdlc / max_score, text="PDLC (relative)")


if __name__ == "__main__":
    run_app()
