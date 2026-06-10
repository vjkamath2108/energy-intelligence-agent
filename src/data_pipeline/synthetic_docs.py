"""Synthetic energy document generator.

Produces ~100 realistic documents across three categories:
  - regulations/  (grid codes, DNO standards, licence conditions)
  - tariffs/      (TNUoS, BSUoS, DUoS charge schedules, Capacity Market)
  - decisions/    (historical dispatch and balancing action records)

Each tariff document is type-specific: TNUoS docs contain only TNUoS
terminology, BSUoS docs only BSUoS, etc. This ensures BM25 can
discriminate between tariff types — required for Precision@5 ≥ 0.70.
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


# ---------------------------------------------------------------------------
# Type-specific tariff content generators
#
# Each generator produces content dense in its own terminology only.
# This lets BM25 rank the correct source prefix highest for each query.
# ---------------------------------------------------------------------------

def _tnuos_content(name: str) -> str:
    """TNUoS-specific tariff schedule — Transmission Network Use of System charges."""
    year = 2024 if "2024" in name else 2025
    region = random.choice(_REGIONS)
    return textwrap.dedent(f"""
    # TNUoS CHARGE SCHEDULE: {name.upper().replace('-', ' ')}
    ## Transmission Network Use of System (TNUoS) Charges — {year}

    Section 1 — TNUoS Generation Tariff Structure

    1.1 The Transmission Network Use of System (TNUoS) generation tariff
    is calculated in accordance with the Transmission Charging Methodology
    Statement (TCMS) approved by Ofgem under the RIIO-T2 price control.
    TNUoS generation charges apply to all transmission-connected generators.

    1.2 TNUoS generation tariff for {region} zone: £{random.uniform(1.5, 8.5):.2f}/kW
    for the {year}/{year % 100 + 1} TNUoS tariff year.
    TNUoS generation charge represents a {random.uniform(-5, 15):.1f}% change
    from the prior TNUoS tariff year, reflecting updated zonal locational signals.

    1.3 TNUoS generation charges are calculated using the transport model
    to reflect the locational cost of connecting generation to the
    transmission system. TNUoS zonal signals encourage generators to locate
    in areas of low transmission constraint.

    Section 2 — TNUoS Demand Tariff and Residual

    2.1 TNUoS demand tariffs recover transmission costs from licensed suppliers
    and large demand customers connected at transmission voltage.
    TNUoS demand charges include a locational component and a residual element.

    2.2 TNUoS demand tariff rate: £{random.uniform(3.0, 12.0):.2f}/kW
    applied to Triad demand (the three highest half-hourly demands in winter).
    TNUoS Triad periods are identified retrospectively by National Grid ESO.

    2.3 TNUoS residual charges recover allowed revenue not allocated
    through locational signals. The TNUoS residual is spread uniformly
    across all transmission-connected demand at £{random.uniform(0.5, 2.5):.4f}/kWh.

    Article 3 — TNUoS Tariff Zones and Methodology

    3.1 The TNUoS tariff year runs from 1 April to 31 March.
    TNUoS charges are published by National Grid ESO each December for
    the following TNUoS tariff year under the TCMS methodology.

    3.2 TNUoS generation tariff zones for {year}:
    Northern TNUoS zone: £{random.uniform(5.0, 12.0):.2f}/kW
    Southern TNUoS zone: £{random.uniform(0.5, 3.0):.2f}/kW
    Midlands TNUoS zone: £{random.uniform(2.0, 6.0):.2f}/kW
    TNUoS zone boundaries reflect transmission constraint costs.

    3.3 TNUoS charges are recovered through the Balancing and Settlement
    Code (BSC). Participants must register their TNUoS zone with the
    relevant Metering System Identifier (MSID) under BSC Section T.

    Clause 4 — TNUoS Embedded Benefits and Exemptions

    4.1 Embedded generators connected below transmission voltage do not
    directly pay TNUoS generation charges but may receive TNUoS embedded
    benefits through reduced demand on the transmission system.

    4.2 TNUoS embedded benefits are calculated as the avoided transmission
    cost attributable to embedded generation displacing transmission flows.
    TNUoS embedded benefit rate: £{random.uniform(0.5, 3.0):.4f}/kWh for {year}.

    4.3 Certain TNUoS exemptions apply under Ofgem decisions for:
    demand response participants, storage facilities, and interconnectors.
    TNUoS exemption applications must be submitted to National Grid ESO
    under the Balancing and Settlement Code procedure BSCP601.
    """)


def _bsuos_content(name: str) -> str:
    """BSUoS-specific tariff schedule — Balancing Services Use of System charges."""
    year = 2024 if "2024" in name else 2025
    quarter = next(
        (q.upper() for q in ("q1", "q2", "q3", "q4") if q in name.lower()), "ANNUAL"
    )
    return textwrap.dedent(f"""
    # BSUoS CHARGE SCHEDULE: {name.upper().replace('-', ' ')}
    ## Balancing Services Use of System (BSUoS) Charges — {quarter} {year}

    Section 1 — BSUoS Charge Overview

    1.1 Balancing Services Use of System (BSUoS) charges recover the costs
    incurred by National Grid ESO in balancing the electricity transmission
    system. BSUoS charges are levied on licensed generators and suppliers
    in proportion to their metered generation or demand in each settlement period.

    1.2 BSUoS charges for {quarter} {year}:
    Forecast BSUoS charge: £{random.uniform(2.5, 7.5):.2f}/MWh
    BSUoS charges are calculated on a half-hourly settlement period basis.
    National Grid ESO publishes BSUoS charges via the Balancing Mechanism
    Reporting Service (BMRS) within two Business Days of each settlement period.

    1.3 BSUoS total system operator costs for {quarter} {year}:
    Balancing services procurement: £{random.uniform(50, 200):.1f}m
    Constraint management costs: £{random.uniform(100, 500):.1f}m
    Reserve and frequency services: £{random.uniform(30, 120):.1f}m
    Total ESO balancing expenditure recovered through BSUoS: £{random.uniform(300, 900):.1f}m

    Section 2 — BSUoS Volatility and Forecasting

    2.1 BSUoS charges are inherently volatile as they reflect real-time
    system operator actions to balance supply and demand.
    BSUoS volatility is driven by: constraint costs, frequency response
    procurement, reserve activation, and interconnector flows.

    2.2 BSUoS volatility adjustment for {quarter} {year}:
    Adjustment factor: {random.uniform(0.85, 1.15):.4f}
    This BSUoS adjustment reflects actual system balancing costs
    versus the forward-looking BSUoS forecast. BSUoS over-recovery
    is refunded; BSUoS under-recovery is collected in subsequent periods.

    2.3 BSUoS charges by service type ({quarter} {year}):
    BSUoS Fast Reserve: £{random.uniform(0.3, 1.2):.2f}/MWh
    BSUoS STOR activation: £{random.uniform(0.1, 0.8):.2f}/MWh
    BSUoS constraint management: £{random.uniform(1.0, 4.0):.2f}/MWh
    BSUoS frequency response: £{random.uniform(0.2, 1.5):.2f}/MWh

    Article 3 — BSUoS Settlement and Billing

    3.1 BSUoS charges are billed monthly through BSC central systems.
    Participants settle BSUoS through their appointed BSC Agent.
    BSUoS invoices are issued within 28 calendar days of month end.

    3.2 BSUoS charge calculation:
    BSUoS payable = metered MWh × BSUoS rate (£/MWh) for each settlement period.
    BSUoS charges apply equally to generation and demand.
    BSUoS dispute process is governed by BSC Section T provisions.

    3.3 BSUoS forecasting service: National Grid ESO publishes a
    rolling 12-month BSUoS forecast each quarter. The BSUoS forward
    price is used by market participants for hedging balancing cost exposure.
    BSUoS forecast for next 12 months: £{random.uniform(3.0, 8.0):.2f}/MWh.

    Clause 4 — BSUoS Cost Drivers and Outlook

    4.1 Key BSUoS cost drivers for {year}:
    Increased renewable generation → higher constraint management costs → higher BSUoS
    Low carbon wind output periods → more BSUoS reserve procurement required
    Interconnector imports → reduced domestic BSUoS balancing requirement

    4.2 BSUoS reduction initiatives by National Grid ESO:
    Pathfinder procurement for BSUoS-reducing services
    Enhanced interconnector coordination to reduce BSUoS costs
    Forward BSUoS cost visibility to improve participant hedging

    4.3 BSUoS charges are reviewed annually by Ofgem under the BSUoS
    methodology statement. Participants may submit BSUoS methodology
    modification proposals through the BSC modification process.
    """)


def _duos_content(name: str) -> str:
    """DUoS-specific tariff schedule — Distribution Use of System charges.

    DUoS and band terminology (Red/Amber/Green) are repeated in every
    paragraph so no single chunk lacks the terms needed for BM25 matching.
    """
    year = 2024 if "2024" in name else 2025
    dno = name.split("duos-charges-")[-1].replace("-2024", "").replace("-2025", "").upper()
    return textwrap.dedent(f"""
    # DUoS CHARGE SCHEDULE: {name.upper().replace('-', ' ')}
    ## Distribution Use of System (DUoS) Charges — {dno} {year}

    Section 1 — DUoS Band Structure and Unit Charges

    1.1 Distribution Use of System (DUoS) charges are levied by the DNO
    on licensed suppliers to recover electricity distribution network costs.
    DUoS charges are structured into three time-of-use bands based on
    system loading: DUoS Red band (peak), DUoS Amber band (shoulder),
    and DUoS Green band (off-peak). DUoS unit charges vary by band.

    1.2 DUoS Red band unit charges — peak demand periods:
    DUoS Red band hours: 16:00–19:00 Monday–Friday (excluding bank holidays).
    DUoS Red band unit charge: {random.uniform(8.0, 25.0):.4f} p/kWh.
    DUoS Red band represents the highest DUoS unit charge period.
    DUoS Red band MPAN fixed charge: {random.uniform(100, 300):.2f} p/day.

    1.3 DUoS Amber band unit charges — shoulder demand periods:
    DUoS Amber band hours: 07:00–16:00 and 19:00–23:00 Monday–Friday.
    DUoS Amber band unit charge: {random.uniform(1.5, 5.0):.4f} p/kWh.
    DUoS Amber band is the intermediate DUoS unit charge period.
    DUoS Amber band MPAN fixed charge: {random.uniform(50, 150):.2f} p/day.

    Section 2 — DUoS Green Band and Standing Charges

    2.1 DUoS Green band unit charges — off-peak demand periods:
    DUoS Green band hours: 23:00–07:00 daily and all-day weekends.
    DUoS Green band unit charge: {random.uniform(0.1, 1.0):.4f} p/kWh.
    DUoS Green band is the lowest DUoS unit charge period.
    DUoS Green band MPAN fixed charge: {random.uniform(10, 50):.2f} p/day.

    2.2 DUoS unit charge summary for {year} — all bands:
    DUoS Red band unit charge: {random.uniform(8.0, 25.0):.4f} p/kWh (peak)
    DUoS Amber band unit charge: {random.uniform(1.5, 5.0):.4f} p/kWh (shoulder)
    DUoS Green band unit charge: {random.uniform(0.1, 1.0):.4f} p/kWh (off-peak)
    DUoS Red:Amber:Green ratio reflects peak network congestion cost allocation.

    2.3 DUoS standing charges per MPAN per day ({year}):
    DUoS fixed charge: {random.uniform(200, 600):.2f} p/day per MPAN.
    DUoS capacity charge: £{random.uniform(1.0, 5.0):.2f}/kVA/month (Red band kVA).
    DUoS excess capacity charge: £{random.uniform(2.0, 8.0):.2f}/kVA/month.

    Article 3 — DUoS Reactive Power and Adjustments

    3.1 DUoS reactive power charges apply where the half-hourly power factor
    at the DUoS metering point falls below 0.95 lagging during Red and Amber band periods.
    DUoS reactive power charge rate: {random.uniform(0.5, 2.0):.4f} p/kVArh.
    DUoS reactive power charges incentivise efficient network utilisation.

    3.2 DUoS annual adjustment mechanism:
    DUoS charges are set annually by the DNO and approved by Ofgem under
    the RIIO-ED2 price control. DUoS Red, Amber and Green band unit charges
    are published in the DNO's annual Statement of Use of System Charges.
    DUoS adjustments ensure full recovery of allowed distribution revenue.

    3.3 DUoS metering requirements:
    Half-hourly metering is required for DUoS time-of-use band billing.
    DUoS half-hourly data submitted to Data Aggregator by D+1 settlement day.
    DUoS band allocation is based on the registered MPAN profile class.
    """)


def _capacity_market_content(name: str) -> str:
    """Capacity Market-specific tariff schedule — T-4 and T-1 agreements."""
    year = 2024 if "2024" in name else 2025
    return textwrap.dedent(f"""
    # CAPACITY MARKET SETTLEMENT SCHEDULE: {name.upper().replace('-', ' ')}
    ## Capacity Market (CM) Agreements and Penalty Rates — {year}

    Section 1 — Capacity Market Agreement Types

    1.1 Capacity Market (CM) agreements are awarded through competitive
    auctions run by National Grid ESO and Ofgem. Capacity Market providers
    holding Capacity Market Agreements (CMAs) receive monthly capacity payments
    in return for committing to be available during Stress Events.

    1.2 Capacity Market T-4 agreements:
    T-4 Capacity Market agreements are awarded four years before the
    delivery year. T-4 Capacity Market agreement rate: £{random.uniform(15, 45):.2f}/kW/year.
    T-4 agreements are available to new-build and refurbishing capacity.
    T-4 Capacity Market obligations run for 1, 3, or 15 years.

    1.3 Capacity Market T-1 agreements:
    T-1 Capacity Market agreements are awarded one year before the
    delivery year. T-1 Capacity Market agreement rate: £{random.uniform(10, 30):.2f}/kW/year.
    T-1 agreements are open to existing capacity and short-lead-time plant.
    T-1 Capacity Market obligations run for 1 year only.

    Section 2 — Capacity Market Stress Events and Penalties

    2.1 A Capacity Market Stress Event is declared by National Grid ESO when
    the System Margin falls below the required level. During a Capacity Market
    Stress Event, all CM agreement holders must deliver their obligated capacity.
    Failure to deliver during a Stress Event triggers Capacity Market penalties.

    2.2 Capacity Market penalty rates for non-delivery during a Stress Event:
    CM Stress Event penalty rate: {random.randint(150, 300)}% of the de-rated clearing price.
    CM T-4 agreement penalty: calculated on T-4 auction clearing price basis.
    CM T-1 agreement penalty: calculated on T-1 auction clearing price basis.
    Capacity Market penalties are capped at the total CM payment received.

    2.3 Capacity Market Stress Event de-rated capacity:
    De-rated capacity for CM obligations: {random.randint(50, 500)} MW per agreement.
    De-rated capacity reflects seasonal availability and forced outage rates.
    Capacity Market de-rating factors are published by National Grid ESO
    in the Capacity Market Rules under CM Rule 4.2.

    Article 3 — Capacity Market Settlement and Billing

    3.1 Capacity Market payments are made monthly to CMA holders.
    CM monthly payment = (annual CM agreement rate × de-rated capacity) / 12.
    T-4 agreement monthly payment: £{random.uniform(1, 5):.2f}m per 100 MW.
    T-1 agreement monthly payment: £{random.uniform(0.5, 3):.2f}m per 100 MW.

    3.2 Capacity Market settlement is administered by the Low Carbon Contracts
    Company (LCCC). Capacity Market invoices are issued monthly.
    T-4 and T-1 agreement payment disputes are resolved under CM Rules Section 6.

    3.3 Capacity Market auction results for {year}:
    T-4 auction clearing price: £{random.uniform(15, 45):.2f}/kW/year.
    T-1 auction clearing price: £{random.uniform(10, 30):.2f}/kW/year.
    Total CM capacity secured: {random.randint(50, 80)} GW de-rated.

    Clause 4 — Capacity Market Obligations and Compliance

    4.1 Capacity Market agreement holders must maintain plant availability
    throughout the CM delivery year. Planned maintenance during Stress Events
    requires prior approval under Capacity Market Rules maintenance provisions.

    4.2 CM prequalification requirements:
    New-build plant must demonstrate CM prequalification by the auction date.
    T-4 and T-1 prequalification tests assess technical capability and
    financial standing. CM prequalification decisions are binding.

    4.3 Capacity Market reforms for {year}:
    Ofgem is reviewing CM penalty rates to ensure Stress Event delivery.
    T-4 agreement duration options under review to support long-term investment.
    CM auction frequency may increase to quarterly from annual under proposals.
    """)


def _generic_tariff_content(name: str) -> str:
    """Generic tariff schedule for non-primary tariff types."""
    year = 2024 if "2024" in name else 2025
    quarter = next(
        (q for q in ("q1", "q2", "q3", "q4") if q in name.lower()), "annual"
    )
    return textwrap.dedent(f"""
    # TARIFF SCHEDULE: {name.upper().replace('-', ' ')}
    ## Effective Period: {year} {quarter.upper()}

    Section 1 — Tariff Overview

    1.1 This tariff schedule sets out charges applicable to participants
    in the GB electricity market for the {year} tariff year.
    Charges are calculated in accordance with the relevant charging methodology
    approved by Ofgem under the applicable licence conditions.

    1.2 Service charges for {year}:
    Base service charge: £{random.uniform(1.5, 8.5):.2f}/kW
    Variable service charge: £{random.uniform(0.5, 3.5):.4f}/kWh
    Standing charge: {random.uniform(100, 600):.2f} p/day per connection point.

    Section 2 — Settlement and Payment Terms

    2.1 Charges are settled through BSC central systems on a monthly basis.
    Invoices are issued within 28 calendar days of the settlement month end.
    Disputes must be lodged within 20 Business Days of invoice receipt.

    2.2 For {quarter.upper()} {year}, forecast service charges reflect
    updated cost-of-service studies and approved regulatory allowances.
    Charges represent a {random.uniform(-5, 15):.1f}% change from the prior year.

    Article 3 — Regulatory Framework

    3.1 These charges are approved by Ofgem under the applicable price control.
    Participants must comply with the relevant charging methodology statement.
    Charges are published annually in December for the following tariff year.

    3.2 Reactive power and ancillary service charges:
    Reactive power charge: £{random.uniform(1.8, 4.5):.4f}/MVArh
    Ancillary service charge: £{random.uniform(0.5, 2.5):.4f}/MWh

    Clause 4 — Contact and Appeals

    4.1 Charges queries should be directed to the relevant system operator.
    Formal appeals are governed by the Balancing and Settlement Code.
    """)


def _tariff_content(name: str) -> str:
    """Dispatch to the correct type-specific generator based on document name."""
    if "tnuos" in name:
        return _tnuos_content(name)
    if "bsuos" in name:
        return _bsuos_content(name)
    if "duos" in name:
        return _duos_content(name)
    if "capacity-market" in name:
        return _capacity_market_content(name)
    return _generic_tariff_content(name)


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
