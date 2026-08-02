"""
Markov model that turns a patient's sequence of per-visit risk states into a
cumulative probability of obstetric hemorrhage across pregnancy + postpartum.

How it works
------------
- Each visit is classified into a risk STATE (LOW / MEDIUM / HIGH) from the
  values recorded at that visit (deterministic, code-only).
- Moving between visits is a Markov step. The state-to-state TRANSITION MATRIX
  is ESTIMATED FROM THE DATA (how often patients move LOW->MEDIUM, etc.).
- Each step also carries a per-state HEMORRHAGE HAZARD (chance of hemorrhaging
  that step). HEMORRHAGE is an absorbing state. The peripartum (labour/postpartum)
  hazards come from a published validation study with real hemorrhage OUTCOMES
  across 261,964 deliveries, keyed to low/medium/high risk tiers (see HAZARD
  below). The antenatal hazards remain small approximate placeholders.
- A patient's hemorrhage probability = 1 - probability of surviving (no
  hemorrhage): each of their own antenatal visits contributes a small antepartum
  hazard (per-patient variation), and the delivery hazard is applied ONCE, keyed
  to their reported risk tier. HEMORRHAGE is the absorbing state.
- The state-to-state transition matrix is also estimated and reported as a
  population-level characterization of how patients move between risk states.

NOTE: the peripartum rates are population averages by risk tier, not validated
against THIS dataset's outcomes (it has none). Treat the output as a
literature-anchored estimate; tune the hazards if better local data exists.
"""

from pipeline.classifier import severity

STATES = ["LOW", "MEDIUM", "HIGH"]

# --- HEMORRHAGE HAZARDS (tunable) ---------------------------------------------
# PERIPARTUM (delivery) hazard = probability of postpartum hemorrhage by risk
# tier, applied ONCE per patient at delivery. Anchored to outcome-labelled rates
# from a validation study of the CMQCC low/medium/high tiers (which map onto our
# states):
#   Ruppel H, Liu VX, Gupta NR, et al. "Validation of Postpartum Hemorrhage
#   Admission Risk Factor Stratification in a Large Obstetrics Population."
#   Am J Perinatol 2020;38(11):1192-1200.  n = 261,964 deliveries.
#   Standard PPH (>= 1000 mL):  low 3.2%, medium 10.5%, high 10.2%.
#   Severe PPH:                 low 0.2%, medium 0.5%, high 1.3%.
# The standard-PPH numbers show medium ~= high, but that reflects a KNOWN
# WEAKNESS of the CMQCC tool (a poor discriminator, AUC ~0.61), not that high-
# and medium-risk patients truly carry equal risk -- the severe-PPH rates show
# HIGH is ~2.6x MEDIUM. So LOW and MEDIUM use the standard-PPH rates directly,
# and HIGH is raised to reflect that gradient. Tune HIGH to taste.
#
# ANTENATAL hazard = per-antenatal-visit antepartum-hemorrhage risk. Antepartum
# hemorrhage is much rarer and not tier-broken-out in the source, so these are
# small approximate values; they give modest patient-to-patient variation from
# each patient's own antenatal trajectory without overriding the delivery tier.
HAZARD = {
    "antenatal":  {"LOW": 0.001, "MEDIUM": 0.003, "HIGH": 0.006},
    "peripartum": {"LOW": 0.032, "MEDIUM": 0.105, "HIGH": 0.150},
}

# Fallback transitions if a state is unseen in the data.
DEFAULT_TRANSITIONS = {
    "LOW":    {"LOW": 0.90, "MEDIUM": 0.08, "HIGH": 0.02},
    "MEDIUM": {"LOW": 0.30, "MEDIUM": 0.50, "HIGH": 0.20},
    "HIGH":   {"LOW": 0.10, "MEDIUM": 0.30, "HIGH": 0.60},
}
# -----------------------------------------------------------------------------

_PERIPARTUM_MARKERS = ("labor", "labour", "postpartum", "postnatal", "sixweek", "six week")


def visit_records(group) -> list[tuple[str, str]]:
    """Return [(state, stage), ...] for a patient's visits in chronological order.
    state is LOW/MEDIUM/HIGH; stage is 'antenatal' or 'peripartum'."""
    recs = []
    cols = list(group.columns)
    for _, row in group.iterrows():
        recs.append((_visit_state(row, cols), _stage(row.get("VisitType"))))
    return recs


def _visit_state(row, cols) -> str:
    severe = borderline = 0
    for c in cols:
        s = severity(c, row.get(c))
        if s == "severe":
            severe += 1
        elif s == "borderline":
            borderline += 1
    if severe >= 2:
        return "HIGH"
    if severe >= 1 or borderline >= 1:
        return "MEDIUM"
    return "LOW"


def _stage(visit_type) -> str:
    t = str(visit_type).strip().lower()
    return "peripartum" if any(m in t for m in _PERIPARTUM_MARKERS) else "antenatal"


def estimate_transitions(state_sequences: list[list[str]]) -> dict:
    """Estimate the state-to-state transition matrix from observed visit sequences."""
    counts = {s: {t: 0 for t in STATES} for s in STATES}
    for seq in state_sequences:
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1

    trans = {}
    for s in STATES:
        total = sum(counts[s].values())
        if total == 0:
            trans[s] = dict(DEFAULT_TRANSITIONS[s])
        else:
            trans[s] = {t: counts[s][t] / total for t in STATES}
    return trans


def hemorrhage_probability(recs: list[tuple[str, str]], overall_tier: str) -> float | None:
    """
    Per-patient probability of hemorrhage over pregnancy + postpartum.

    - Each of the patient's own antenatal visits contributes a small antepartum-
      hemorrhage hazard based on that visit's state (this is where patient-to-
      patient variation within a tier comes from).
    - The delivery (peripartum) hemorrhage hazard is applied ONCE, keyed to the
      patient's reported overall risk tier (LOW/MEDIUM/HIGH) so the number tracks
      the actual assessed severity rather than a separate per-visit guess.
    """
    if overall_tier not in STATES:
        return None

    survival = 1.0
    for state, stage in recs:
        if stage == "antenatal":
            survival *= (1 - HAZARD["antenatal"][state])

    survival *= (1 - HAZARD["peripartum"][overall_tier])   # the single delivery event
    return round(1 - survival, 4)
