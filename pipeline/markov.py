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
- A patient's hemorrhage probability is the chance of being absorbed into
  HEMORRHAGE: starting from their reported risk tier (LOW/MEDIUM/HIGH), the tier
  is projected forward to delivery with the transition matrix, and the peripartum
  (delivery) hazard is applied ONCE. This keeps the probability consistent with
  the reported category and monotonic (LOW <= MEDIUM <= HIGH). Because the
  transition matrix captures that HIGH patients persist in HIGH while MEDIUM
  rarely reach it, the projection separates the tiers more than the near-equal
  raw peripartum rates would on their own.

NOTE: the peripartum rates are population averages by risk tier, not validated
against THIS dataset's outcomes (it has none). Treat the output as a
literature-anchored estimate; tune the hazards if better local data exists.
"""

from pipeline.classifier import severity

STATES = ["LOW", "MEDIUM", "HIGH"]

# --- HEMORRHAGE HAZARDS (tunable) ---------------------------------------------
# Per-step probability of hemorrhage given the visit's risk state.
#
# PERIPARTUM (labour/postpartum) hazards use REAL outcome-labelled rates from a
# validation study of the CMQCC low/medium/high admission risk tiers, which map
# directly onto our states:
#   Ruppel H, Liu VX, Gupta NR, et al. "Validation of Postpartum Hemorrhage
#   Admission Risk Factor Stratification in a Large Obstetrics Population."
#   Am J Perinatol 2020;38(11):1192-1200.  n = 261,964 deliveries.
#   Standard PPH (blood loss >= 1000 mL):  low 3.2%, medium 10.5%, high 10.2%.
#   (Severe-PPH alternative, cleaner gradient: low 0.2%, medium 0.5%, high 1.3%.)
# The source found medium ~= high for standard PPH (the tool is a weak
# discriminator, AUC ~0.61); HIGH is nudged just above MEDIUM here so the model
# stays monotonic. To use the severe-PPH definition instead, swap in
# {LOW: 0.002, MEDIUM: 0.005, HIGH: 0.013}.
#
# ANTENATAL hazards represent antepartum hemorrhage, which is much rarer than PPH
# and is NOT broken out by these tiers in the source, so these remain small,
# approximate placeholders — tune if a better source is available.
HAZARD = {
    "antenatal":  {"LOW": 0.001, "MEDIUM": 0.005, "HIGH": 0.010},
    "peripartum": {"LOW": 0.032, "MEDIUM": 0.105, "HIGH": 0.110},
}

# If a patient hasn't reached labour/postpartum yet, project this many more
# antenatal visits before the (always-added) postpartum step.
FUTURE_ANTENATAL_STEPS = 1

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


def tier_probabilities(transitions: dict) -> dict:
    """
    The hemorrhage probability for each risk tier, computed ONCE from the Markov
    chain and reused for every patient of that tier. This keeps the probability
    consistent with the reported LOW/MEDIUM/HIGH category and guarantees
    LOW <= MEDIUM <= HIGH (no more HIGH patients scoring below MEDIUM ones).
    """
    vals = {tier: _absorption_probability(tier, transitions) for tier in STATES}
    # Enforce monotonicity in case a quirky estimated matrix would break it.
    vals["MEDIUM"] = max(vals["MEDIUM"], vals["LOW"])
    vals["HIGH"] = max(vals["HIGH"], vals["MEDIUM"])
    return {k: round(v, 4) for k, v in vals.items()}


def _absorption_probability(start_tier: str, transitions: dict) -> float:
    """
    Probability of being absorbed into HEMORRHAGE for a patient currently in
    `start_tier`: project the tier forward toward delivery with the transition
    matrix (antenatal steps, each carrying a small antepartum-hemorrhage hazard),
    then apply the peripartum (delivery) hemorrhage hazard ONCE.
    """
    dist = {s: 0.0 for s in STATES}
    dist[start_tier] = 1.0

    survival = 1.0
    for _ in range(FUTURE_ANTENATAL_STEPS):
        aph = sum(dist[s] * HAZARD["antenatal"][s] for s in STATES)
        survival *= (1 - aph)
        dist = _advance(dist, transitions)

    pph = sum(dist[s] * HAZARD["peripartum"][s] for s in STATES)   # applied once
    survival *= (1 - pph)
    return 1 - survival


def _advance(dist: dict, transitions: dict) -> dict:
    nxt = {s: 0.0 for s in STATES}
    for s, p in dist.items():
        for t, q in transitions[s].items():
            nxt[t] += p * q
    return nxt
