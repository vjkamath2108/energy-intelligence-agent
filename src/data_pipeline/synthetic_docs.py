"""Synthetic energy document generator.

Produces ~100 realistic documents across three categories:
  - regulations/  (grid codes, DNO standards, licence conditions)
  - tariffs/      (TNUoS, BSUoS, DUoS charge schedules)
  - decisions/    (historical dispatch and balancing action records)

Each document is a dict with keys: source, content, category.
"""

import random
import textwrap
from typing import TypedDict

random.seed(42)


class SyntheticDocument(TypedDict):
    """A single synthetic energy document."""

    source: str
    content: str
    category: str


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_REGULATION_NAMES = [
    "dno-p28", "grid-code-section-4", "grid-code-section-6",
    "ner-licence-condition-c17", "ner-licence-condition-d14",
    "eso-balancing-code-bc1", "eso-balancing-code-bc3",
    "gg-o6-dispatch-principles", "sqss-engineering-recommendation",
    "p305-modification", "p354-modification", "cusc-section-11",
    "cusc-section-14", "bsc-section-v", "bsc-section-x",
    "connection-conditions-cc6", "transmission-licence-special-conditions",
    "distribution-code-dcp228", "ofgem-decision-ref-2024-031",
    "ner-standard-licence-conditions-2024",
]

_TARIFF_NAMES = [
    "tnuos-charges-2024", "tnuos-charges-2025",
    "bsuos-rates-q1-2024", "bsuos-rates-q2-2024",
    "bsuos-rates-q3-2024", "bsuos-rates-q4-2024",
    "duos-charges-enwl-2024", "duos-charges-ssen-2024",
    "duos-charges-ukpn-2024", "duos-charges-npg-2024",
    "capacity-market-settlement-2024", "ro-obligation-level-2024",
    "cfd-strike-prices-ar5", "cfd-strike-prices-ar6",
    "embedded-benefits-statement-2024", "reactive-power-tariff-2024",
    "black-start-service-charges-2024", "firm-frequency-response-tariff",
    "mandatory-frequency-response-tariff", "enhanced-frequency-response-tariff",
]

_DECISION_NAMES = [
    "dispatch-2024-01-15", "dispatch-2024-02-03", "dispatch-2024-03-22",
    "dispatch-2024-04-11", "dispatch-2024-05-07", "dispatch-2024-06-19",
    "balancing-action-refs-q1", "balancing-action-refs-q2",
    "constraint-management-2024-north", "constraint-management-2024-south",
    "redispatch-event-2024-0312", "redispatch-event-2024-0517",
    "bm-unit-acceptance-2024-jan", "bm-unit-acceptance-2024-apr",
    "reserve-activation-2024-q1", "reserve-activation-2024-q2",
    "emergency-instruction-2024-001", "emergency-instruction-2024-002",
    "post-event-report-2024-0601", "post-event-report-2024-0823",
    "balancing-mechanism-report-h1-2024", "ngeso-control-room-log-2024-q1",
]

_VOLTAGE_LEVELS = ["11 kV", "33 kV", "132 kV", "275 kV", "400 kV"]
_SERVICES = ["FFR", "DC Low", "DC High", "BM", "STOR", "Fast Reserve", "Black Start"]
_REGIONS = ["North Scotland", "South Scotland", "North West", "North East",
            "Yorkshire", "East Midlands", "West Midlands", "East", "London",
            "South East", "South West", "Wales"]
_UNITS = ["DRAX1G", "HUMR1G", "TORN1G", "STTG1G", "PEHE1G",
          "RATS1G", "ABTH1G", "FAWN1G", "COTPS1", "FERR1G"]


# ---------------------------------------------------------------------------
# Per-category content generators
# ---------------------------------------------------------------------------

def _regulation_content(name: str) -> str:
    """Generate a realistic regulation document."""
    parts = [
        f"# {name.upper().replace('-', ' ')}\n",
        f"## Regulatory Reference: {name}\n",
        textwrap.dedent(f"""
        Section 1 — Purpose and Scope

        1.1 This document establishes the technical and operational requirements
        applicable to all transmission-connected generating units and demand-side
        participants operating within the GB transmission system.

        1.2 Compliance with these requirements is mandatory under the terms of the
        relevant Grid Code and Balancing and Settlement Code (BSC) as administered
        by National Grid ESO (NGESO).

        1.3 Participants must demonstrate compliance with voltage regulation
        requirements at the connection point, maintaining reactive power output
        within the capability diagram defined in Appendix A.

        Section 2 — Voltage and Reactive Power Requirements

        2.1 All transmission-connected generators must maintain operation within
        the voltage range {random.choice(_VOLTAGE_LEVELS)} to {random.choice(_VOLTAGE_LEVELS)}
        as measured at the grid supply point (GSP).

        2.2 The reactive power capability requirement is defined as the ability to
        absorb or generate reactive power in accordance with the P–Q capability
        diagram specified in Schedule 1.

        2.3 Where a generating unit cannot comply with the reactive capability
        requirements, the participant shall notify the System Operator no later
        than 15 Business Days before the scheduled outage or modification date.

        Article 3 — Frequency Response Obligations

        3.1 Primary frequency response shall be delivered within 10 seconds of
        the frequency deviation event and maintained for a minimum of 20 seconds.

        3.2 Secondary frequency response shall be delivered within 30 seconds
        and maintained until the System Operator issues a stand-down instruction.

        3.3 Participants providing {random.choice(_SERVICES)} services must ensure
        their control systems are capable of automated response without manual
        intervention.

        Clause 4 — Dispatch Instructions

        4.1 All generating units participating in the Balancing Mechanism (BM)
        shall respond to dispatch instructions issued by the System Operator
        within the gate closure period.

        4.2 The nominated de-rated capacity for {random.choice(_REGIONS)} shall
        be updated in accordance with BSC Section V provisions.

        4.3 Generating unit {random.choice(_UNITS)} is subject to the following
        additional obligations as specified under this licence condition.

        Section 5 — Metering and Data Submission

        5.1 Half-hourly metered data must be submitted to the Data Aggregator
        by D+1 (next working day after the settlement day).

        5.2 All submitted data must conform to the MSID format specified in
        BSC Procedure BSCP509.

        5.3 Participants must retain metering records for a minimum of 6 years
        from the date of the settlement period to which they relate.

        Appendix A — Capability Diagram Parameters

        The minimum reactive power absorption capability at rated MW output
        shall not be less than {random.randint(20, 40)}% of the rated MVA.
        The minimum reactive power generation capability at rated MW output
        shall not be less than {random.randint(20, 35)}% of the rated MVA.
        At zero MW output, participants must maintain reactive absorption capability
        equivalent to at least {random.randint(10, 25)}% of rated MVA.
        """),
    ]
    return "\n".join(parts)


def _tariff_content(name: str) -> str:
    """Generate a realistic tariff schedule document."""
    year = 2024 if "2024" in name else 2025
    quarter = next(
        (q for q in ("q1", "q2", "q3", "q4") if q in name.lower()), "annual"
    )
    return textwrap.dedent(f"""
    # TARIFF SCHEDULE: {name.upper().replace('-', ' ')}
    ## Effective Period: {year} {quarter.upper()}

    Section 1 — TNUoS Tariff Structure

    1.1 The Transmission Network Use of System (TNUoS) tariff is calculated in
    accordance with the methodology set out in the Transmission Charging Methodology
    Statement (TCMS) and approved by Ofgem.

    1.2 Generation tariffs for {random.choice(_REGIONS)} zone are set at
    £{random.uniform(1.5, 8.5):.2f}/kW for the {year}/{year % 100 + 1} tariff year,
    representing a {random.uniform(-5, 15):.1f}% change from the prior year.

    1.3 Demand tariffs reflect the zonal locational signals as calculated under
    the transport model. The residual element is recovered uniformly across
    all transmission-connected demand.

    Section 2 — BSUoS Charges

    2.1 Balancing Services Use of System (BSUoS) charges recover the costs
    incurred by National Grid ESO in balancing the transmission system.

    2.2 For {quarter.upper()} {year}, the forecast BSUoS charge is
    £{random.uniform(2.5, 7.5):.2f}/MWh, calculated on a 12-month forward-looking basis.

    2.3 The volatility adjustment factor applied to BSUoS in this period is
    {random.uniform(0.85, 1.15):.4f}, reflecting {random.choice(['higher', 'lower'])}
    than forecast constraint costs in {random.choice(_REGIONS)}.

    Article 3 — DUoS Charge Structure

    3.1 Distribution Use of System (DUoS) charges are banded into three time
    periods: Red (peak), Amber (shoulder), and Green (off-peak).

    3.2 Red band unit charges: {random.uniform(8.0, 25.0):.4f} p/kWh
        Amber band unit charges: {random.uniform(1.5, 5.0):.4f} p/kWh
        Green band unit charges: {random.uniform(0.1, 1.0):.4f} p/kWh

    3.3 Fixed charges: {random.uniform(200, 600):.2f} p/day per MPAN.

    Clause 4 — Capacity Market Settlement

    4.1 Capacity Market (CM) payments are made to providers holding
    Capacity Market Agreements (CMAs) at the following rates for {year}:
    T-4 agreements: £{random.uniform(15, 45):.2f}/kW/year
    T-1 agreements: £{random.uniform(10, 30):.2f}/kW/year

    4.2 Penalty rates for non-delivery during a Stress Event are set at
    {random.randint(150, 300)}% of the de-rated price for the relevant auction year.

    Section 5 — Reactive Power Tariff

    5.1 Reactive power charges apply to generating units operating outside
    their obligatory reactive power range (ORPR) as defined in Grid Code OC5.

    5.2 The reactive power charge rate is £{random.uniform(1.8, 4.5):.4f}/MVArh
    for absorption beyond the ORPR, and £{random.uniform(2.0, 5.0):.4f}/MVArh
    for generation beyond the ORPR.
    """)


def _decision_content(name: str) -> str:
    """Generate a realistic dispatch decision record."""
    unit = random.choice(_UNITS)
    mw_offer = random.randint(100, 800)
    mw_accepted = random.randint(50, mw_offer)
    price = random.uniform(45.0, 250.0)
    return textwrap.dedent(f"""
    # DISPATCH / BALANCING ACTION RECORD
    ## Reference: {name.upper().replace('-', ' ')}

    Section 1 — Event Summary

    1.1 This record documents the dispatch instruction issued by National Grid ESO
    to BM Unit {unit} under the Balancing Mechanism provisions of the BSC.

    1.2 The instruction was issued in response to a system imbalance event
    in the {random.choice(_REGIONS)} region requiring {mw_accepted} MW of
    {'upward' if random.random() > 0.5 else 'downward'} balancing action.

    1.3 Gate closure time: Settlement Period {random.randint(1, 48)},
    Gate closure volume accepted: {mw_accepted} MW at £{price:.2f}/MWh.

    Section 2 — Offer Stack at Time of Dispatch

    2.1 The accepted offer from {unit} was the
    {random.choice(['cheapest available', 'only available', 'most accessible'])}
    offer in the {random.choice(_REGIONS)} constraint zone at the time of dispatch.

    2.2 Full offer stack (top 5 accepted):
    - {random.choice(_UNITS)}: {random.randint(50, 500)} MW @ £{random.uniform(40, 200):.2f}/MWh
    - {random.choice(_UNITS)}: {random.randint(50, 400)} MW @ £{random.uniform(50, 220):.2f}/MWh
    - {unit}: {mw_accepted} MW @ £{price:.2f}/MWh
    - {random.choice(_UNITS)}: {random.randint(30, 300)} MW @ £{random.uniform(60, 250):.2f}/MWh
    - {random.choice(_UNITS)}: {random.randint(20, 200)} MW @ £{random.uniform(70, 280):.2f}/MWh

    Article 3 — Constraint Background

    3.1 The constraint was triggered by an outage on the
    {random.choice(_VOLTAGE_LEVELS)} transmission circuit in {random.choice(_REGIONS)},
    reducing available import capacity by {random.randint(200, 1200)} MW.

    3.2 The Security and Quality of Supply Standard (SQSS) N-1 requirement
    necessitated balancing action to maintain system security.

    Clause 4 — Financial Settlement

    4.1 Energy accepted in BM: {mw_accepted} MWh
    Imbalance price (SBP): £{random.uniform(45, 300):.2f}/MWh
    Imbalance price (SSP): £{random.uniform(30, 200):.2f}/MWh
    Total BM cost to ESO: £{mw_accepted * price:,.2f}

    4.2 The action is recoverable through BSUoS charges in accordance
    with BSC Section T provisions.

    Section 5 — Post-Event Review

    5.1 System frequency deviation: {random.uniform(-0.3, 0.3):.3f} Hz
    5.2 Duration of constraint: {random.randint(15, 240)} minutes
    5.3 Constraint cost: £{random.uniform(10000, 500000):,.2f}
    5.4 Review outcome: {random.choice([
        'No further action required.',
        'Flagged for network investment appraisal.',
        'Referred to Ofgem for licence compliance review.',
        'Included in quarterly constraint cost report.',
    ])}
    """)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_documents() -> list[SyntheticDocument]:
    """Generate ~100 synthetic energy documents across three categories.

    Returns:
        List of dicts with keys source, content, and category.
        Distribution: ~33 regulations, ~33 tariffs, ~34 decisions.
    """
    docs: list[SyntheticDocument] = []

    reg_names = random.sample(_REGULATION_NAMES, min(34, len(_REGULATION_NAMES)))
    for name in reg_names:
        docs.append({
            "source": f"regulations/{name}.pdf",
            "content": _regulation_content(name),
            "category": "regulations",
        })

    tar_names = random.sample(_TARIFF_NAMES, min(33, len(_TARIFF_NAMES)))
    for name in tar_names:
        docs.append({
            "source": f"tariffs/{name}.pdf",
            "content": _tariff_content(name),
            "category": "tariffs",
        })

    dec_names = random.sample(_DECISION_NAMES, min(33, len(_DECISION_NAMES)))
    for name in dec_names:
        docs.append({
            "source": f"decisions/{name}.pdf",
            "content": _decision_content(name),
            "category": "decisions",
        })

    return docs


if __name__ == "__main__":
    docs = generate_documents()
    cats: dict[str, int] = {}
    total_words = 0
    for d in docs:
        cats[d["category"]] = cats.get(d["category"], 0) + 1
        total_words += len(d["content"].split())
    print(f"Generated {len(docs)} documents — {cats}")
    print(f"Total words: {total_words:,}  avg: {total_words // len(docs):,}/doc")
