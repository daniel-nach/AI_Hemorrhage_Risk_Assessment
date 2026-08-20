"""
Markov model: predict how a patient's key CLINICAL states evolve to delivery,
then derive the hemorrhage probability from the predicted delivery state.

This models the actual conditions that progress during pregnancy and drive
hemorrhage risk -- not an abstract LOW/MEDIUM/HIGH chain:
  - Blood pressure / hypertensive disease  (Normotensive -> Gestational HTN ->
    Preeclampsia -> Severe Preeclampsia)
  - Anemia                                 (None -> Mild -> Moderate -> Severe)

For each patient we read the CURRENT state of each condition from their latest
readings, project it forward to delivery using transition probabilities taken
from published cohort studies, and map the predicted delivery-state distribution
to a postpartum-hemorrhage probability (via CMQCC risk tiers + the Ruppel et al.
PPH rates). Every transition probability comes from a study -- see CITATIONS.md.
Nothing here is estimated from the project's (synthetic, outcome-free) dataset.

--------------------------------------------------------------------------------
TIME DISCRETIZATION  (professor approved option B; option A kept for reference)

The cited progression rates are CUMULATIVE over pregnancy, so they must be turned
into "risk over the pregnancy the patient has left." Two approaches were weighed:

  (A) TRIMESTER STAGING -- split pregnancy into trimesters and apply per-trimester
      transition probabilities. Would use the same cohort rates broken out by
      stage, e.g. anemia prevalence 11.8% (T1) -> 28.8% (T3)
      [BMC Public Health 2024, PMC11034068], and HDP onset only after 20 weeks
      [ACOG PB 222]. More faithful to when events happen, but needs per-stage
      rates the sources don't always report.

  (B) REMAINING-WINDOW SCALING  <-- CHOSEN. Scale each cumulative rate by the
      fraction of the at-risk window the patient has left, using gestational age
      (from fundal height). Assuming a constant per-week hazard, a cumulative
      probability p over a window of W weeks gives, over w remaining weeks:
            p_remaining = 1 - (1 - p) ** (w / W)
      Simpler and needs only the cumulative rates the sources do report; the
      constant-hazard assumption is the one modelling assumption on top of the
      cited numbers.

Both approaches draw on the same progression studies; (A) is documented above so
we can switch later without re-deriving it.
--------------------------------------------------------------------------------

SCOPE NOTE: only blood pressure and anemia are modelled, because those are the
conditions with published pregnancy-progression rates. Other stats (pulse,
platelets, O2) lack cited transition data and are therefore left to the
categorical risk assessment, not this probability. The probability here is
specifically the hypertensive-/anemia-progression-driven hemorrhage risk.
"""

from pipeline.classifier import _f, anemia_severity

# --- Timing assumptions (option B) --- see CITATIONS.md -----------------------
TERM_WEEKS = 40.0
# At-risk window START (weeks). Hypertensive disorders of pregnancy are defined
# only after 20 weeks [ACOG Practice Bulletin 222]; anemia onset is tracked from
# the end of the first trimester (~12 wks), matching the 1st->3rd trimester
# cohort measure [BMC Public Health 2024].
AT_RISK_START = {"bp": 20.0, "anemia": 12.0}
DEFAULT_GA = 28.0   # fallback gestational age when fundal height is missing

# --- Blood-pressure / hypertensive-disease chain ------------------------------
# States ordered by severity.
BP_STATES = ["NORMOTENSIVE", "GEST_HTN", "PREECLAMPSIA", "SEVERE_PE"]
# Cumulative probability (over the remaining at-risk window) of progressing FROM
# each state TO a worse one. Anything not listed = stays in the current state.
# Setup / sources for each assumption:
#   NORMOTENSIVE -> GEST_HTN 14.6% and -> PREECLAMPSIA 2.1%
#     From a large prospective cohort where, over pregnancy, 80.0% remained
#     normotensive, 14.6% developed gestational hypertension and 2.1% developed
#     preeclampsia. [Hypertensive Disorders of Pregnancy, PMC7720658]
#   GEST_HTN -> PREECLAMPSIA 17.1%
#     Northern Ethiopia prospective cohort (41/240 = 17.1%); 15-25% across
#     studies. [PMC8008690]
#   PREECLAMPSIA -> SEVERE_PE 5.0%
#     Cohort of mild hypertensive disorders progressing to severe features
#     (18/359 ~= 5%). [PMC10672209]
BP_CUM_TRANSITIONS = {
    "NORMOTENSIVE": {"GEST_HTN": 0.146, "PREECLAMPSIA": 0.021},
    "GEST_HTN":     {"PREECLAMPSIA": 0.171},
    "PREECLAMPSIA": {"SEVERE_PE": 0.050},
    "SEVERE_PE":    {},
}

# --- Anemia chain -------------------------------------------------------------
ANEMIA_STATES = ["NONE", "MILD", "MODERATE", "SEVERE"]
# Setup / sources:
#   NONE -> (new) anemia 7.9%
#     Of women without anemia in the first trimester, 7.9% developed anemia by
#     the third trimester. [multi-center cohort PMC11034068; early-Hb-predicts-
#     late-anemia PMC8876051]. New-onset anemia is modelled as MILD (most new
#     anemia is mild).
#   Worsening of an already-anemic patient (MILD -> MODERATE, etc.) is NOT
#     reliably quantified in the literature, so those states are HELD (no cited
#     worsening transition). Flagged in CITATIONS.md.
ANEMIA_CUM_TRANSITIONS = {
    "NONE":     {"MILD": 0.079},
    "MILD":     {},
    "MODERATE": {},
    "SEVERE":   {},
}

# --- Predicted state -> hemorrhage risk tier -> PPH probability ----------------
# Preeclampsia (especially severe) and moderate/severe anemia are established PPH
# risk factors and CMQCC high-risk criteria [CMQCC Toolkit V3]. Each condition
# maps to a risk tier; a patient's tier is the worse of the two. Tiers then use
# the validated PPH rates [Ruppel et al. 2020] (HIGH raised per the severe-PPH
# gradient -- see CITATIONS.md).
BP_TIER = {"NORMOTENSIVE": "low", "GEST_HTN": "medium", "PREECLAMPSIA": "high", "SEVERE_PE": "high"}
ANEMIA_TIER = {"NONE": "low", "MILD": "medium", "MODERATE": "high", "SEVERE": "high"}
PPH_RATE = {"low": 0.032, "medium": 0.105, "high": 0.150}
_TIER_ORDER = {"low": 0, "medium": 1, "high": 2}


def estimate_ga(fundal_height) -> float | None:
    """Gestational age in weeks from fundal height (cm ~= weeks, McDonald's rule)."""
    fh = _f(fundal_height)
    if fh is None or not (10 <= fh <= 45):
        return None
    return fh


def current_bp_state(sysbp, diabp, urine_protein) -> str | None:
    """Classify the patient's current hypertensive-disease state from latest readings.
    Thresholds per ACOG PB 222 (HTN >=140/90, severe >=160/110, proteinuria >=2+)."""
    s, d = _f(sysbp), _f(diabp)
    if s is None and d is None:
        return None
    prot = str(urine_protein).strip().lower()
    signif_prot = prot in ("2+", "3+", "4+")
    severe_prot = prot in ("3+", "4+")
    htn = (s is not None and s >= 140) or (d is not None and d >= 90)
    severe_htn = (s is not None and s >= 160) or (d is not None and d >= 110)
    if not htn and not signif_prot:
        return "NORMOTENSIVE"
    if severe_htn or severe_prot:
        return "SEVERE_PE"
    if htn and signif_prot:
        return "PREECLAMPSIA"
    return "GEST_HTN"


def current_anemia_state(hb) -> str | None:
    """Current anemia state from latest Hb. Uses the shared WHO banding in
    classifier.anemia_severity so this and the classifier can never disagree."""
    if _f(hb) is None:
        return None
    sev = anemia_severity(hb)   # 'severe' / 'moderate' / 'mild' / None
    return {"severe": "SEVERE", "moderate": "MODERATE", "mild": "MILD"}.get(sev, "NONE")


def hemorrhage_probability(bp_state, anemia_state, ga) -> float | None:
    """
    Project the patient's BP and anemia states forward to delivery, then return the
    expected postpartum-hemorrhage probability over the predicted delivery states.
    """
    if bp_state is None and anemia_state is None:
        return None

    bp_dist = _project(bp_state, BP_CUM_TRANSITIONS, BP_STATES, "bp", ga) if bp_state else {None: 1.0}
    an_dist = _project(anemia_state, ANEMIA_CUM_TRANSITIONS, ANEMIA_STATES, "anemia", ga) if anemia_state else {None: 1.0}

    # Expected PPH over the joint delivery-state distribution (BP and anemia
    # treated as independent -- a documented simplifying assumption).
    expected = 0.0
    for b, pb in bp_dist.items():
        for a, pa in an_dist.items():
            bt = BP_TIER.get(b, "low")
            at = ANEMIA_TIER.get(a, "low")
            tier = bt if _TIER_ORDER[bt] >= _TIER_ORDER[at] else at
            expected += pb * pa * PPH_RATE[tier]
    return round(expected, 4)


def _project(state, cum_transitions, states, chain, ga) -> dict:
    """One-step Markov projection to delivery, with option-B time scaling."""
    frac = _remaining_fraction(chain, ga)
    dist = {st: 0.0 for st in states}
    moved = 0.0
    for target, p_cum in cum_transitions[state].items():
        p = 1 - (1 - p_cum) ** frac      # option B: constant-hazard time scaling
        dist[target] += p
        moved += p
    dist[state] += 1 - moved              # remainder stays in the current state
    return dist


def _remaining_fraction(chain, ga) -> float:
    """Fraction of the at-risk window still ahead of the patient (0..1)."""
    if ga is None:
        ga = DEFAULT_GA
    window = TERM_WEEKS - AT_RISK_START[chain]
    if window <= 0:
        return 0.0
    return min(1.0, max(0.0, (TERM_WEEKS - ga) / window))
