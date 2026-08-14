# Citations & Sources

This document lists the source behind every clinical number in the pipeline, and
is honest about what is directly sourced, what follows a standard convention, and
what is a project design choice. Inline `# source:` comments in the code point
back here.

Summary of provenance:

- **Directly sourced & verified** — the Markov hemorrhage rates and the
  low/medium/high risk framework.
- **Standard authoritative thresholds** — anemia (WHO), hypertension &
  proteinuria (ACOG), BMI (WHO). Coded from convention; the authoritative source
  is confirmed here.
- **General clinical convention** — routine vitals (pulse, O2 sat, temperature,
  respiratory rate, platelets). No single citable paper.
- **Project design choices (NOT sourced)** — the "normal, upper/lower end" edge
  bands, the HIGH peripartum hazard, the antepartum hazards, the transition
  matrix (estimated from data), and the triage/pipeline structure.

---

## 1. Directly sourced & verified

### Postpartum hemorrhage rates by risk tier — `pipeline/markov.py` (HAZARD)
Ruppel H, Liu VX, Gupta NR, et al. **"Validation of Postpartum Hemorrhage
Admission Risk Factor Stratification in a Large Obstetrics Population."**
*American Journal of Perinatology* 2020;38(11):1192–1200. n = 261,964 deliveries.
- Standard PPH (≥1000 mL blood loss): low **3.2%**, medium **10.5%**, high **10.2%**.
- Severe PPH: low **0.2%**, medium **0.5%**, high **1.3%**.
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7688483/

Used for the peripartum (delivery) hazards: LOW and MEDIUM take the standard-PPH
rates directly. See §4 for the HIGH value, which is a project extrapolation.

### Low / medium / high risk framework — the risk states our model mirrors
CMQCC (California Maternal Quality Care Collaborative). **Improving Health Care
Response to Obstetric Hemorrhage Toolkit, Version 3.0** (2022), Appendix K:
Obstetric Hemorrhage Risk Factor Assessment Screen.
- https://www.cmqcc.org/resource/improving-health-care-response-obstetric-hemorrhage-toolkit-version-30
- https://www.cmqcc.org/resource/ob-hemorrhage-toolkit-v30-appendix-k-obstetric-hemorrhage-risk-factor-assessment-screen

---

## 2. Standard authoritative thresholds — `pipeline/classifier.py`, `pipeline/loader.py`

### Hemoglobin / anemia in pregnancy — `_hb()`
**WHO** haemoglobin classification for anaemia in pregnancy: anemia <11 g/dL;
mild 10–10.9, moderate 7–9.9, severe <7.0.
- https://dhsprogram.com/data/Guide-to-DHS-Statistics/Anemia_Status.htm
- FIGO 2025 good-practice recommendations (discuss WHO thresholds):
  https://obgyn.onlinelibrary.wiley.com/doi/full/10.1002/ijgo.70529

### Blood pressure & proteinuria — `_sysbp()`, `_diabp()`, `_urine_protein()`
**ACOG** Practice Bulletin No. 222, *Gestational Hypertension and Preeclampsia*:
hypertension ≥140/90 mmHg; severe-range ≥160/110 mmHg; proteinuria dipstick ≥2+
(or ≥300 mg/24h).
- https://www.preeclampsia.org/frontend/assets/img/advocacy_resource/Gestational_Hypertension_and_Preeclampsia_ACOG_Practice_Bulletin,_Number_222_1605448006.pdf
- Hypertension in Pregnancy, StatPearls / NIH: https://www.ncbi.nlm.nih.gov/books/NBK430839/

### BMI classification — `_bmi_category()`
**WHO** BMI categories: underweight <18.5, normal 18.5–24.9, overweight 25–29.9,
obese ≥30.
- https://www.worldobesity.org/about/about-obesity/obesity-classification

### Fundal height ≈ gestational age — used in the prompt guidance
Standard obstetric rule of thumb (McDonald's rule): symphysis-fundal height in cm
≈ gestational age in weeks between ~24–36 weeks.

---

## 3. General clinical convention (no single citable source)

Coded from routine clinical vitals knowledge — `pipeline/classifier.py`:

- **Pulse** (`_pulse`): tachycardia >100–110 bpm (resting HR runs higher in
  pregnancy); bradycardia <50.
- **Oxygen saturation** (`_o2`): normal ≥95%, concerning <92%.
- **Temperature** (`_temp`): fever ≥100.4 °F (38 °C).
- **Respiratory rate** (`_resp`): normal 12–20/min.
- **Platelets** (`_platelet`): thrombocytopenia <150,000; severe <100,000
  (the <100k also aligns with HELLP-syndrome criteria).
- **Blood loss** (`_blood_loss`): PPH defined as ≥500 mL (vaginal) / ≥1000 mL;
  the >500 threshold here is approximate.

---

## 4. Project design choices — NOT from an external source

These should be reviewed / tuned by the clinician:

- **HIGH peripartum hazard = 0.15** (`pipeline/markov.py`). An *extrapolation*,
  not a published figure. The Ruppel standard-PPH rates put medium ≈ high (10.5 vs
  10.2), but that reflects a documented weakness of the CMQCC tool as a
  discriminator (AUC ~0.61); the same study's *severe*-PPH rates show HIGH ≈ 2.6×
  MEDIUM. HIGH was raised to 0.15 to reflect that true gradient.
- **Antepartum per-visit hazards** (LOW 0.001, MEDIUM 0.003, HIGH 0.006,
  `pipeline/markov.py`). Small approximate placeholders; antepartum hemorrhage is
  rarer than PPH and not tier-broken-out in the source.
- **"normal, upper/lower end" edge bands** (`pipeline/classifier.py`), e.g. SysBP
  130–139, DiaBP 85–89, pulse 95–100. A design device for the aggravating-factor
  logic — they flag near-threshold values, and are NOT clinical categories.
- **State transition matrix** (`pipeline/markov.py`). Estimated from the dataset's
  visit-to-visit state changes, not from literature.
- **Triage rules and overall pipeline structure** (`pipeline/loader.py`,
  `main.py`). Engineering design.

---

## Known deviations from the cited sources

- **Hemoglobin bands run slightly lenient vs WHO.** The code labels Hb 9–9.9 as
  "mild" (WHO: moderate) and 10–10.9 as "normal, lower end" (WHO: mild anemia). To
  conform strictly to WHO, shift `_hb()` in `pipeline/classifier.py` so 7–9.9 is
  moderate and 10–10.9 is mild.
- **Trimester-specific anemia cutoffs are not applied.** WHO's 2024 update lowers
  the second-trimester anemia cutoff to 10.5 g/dL; the code uses the single 11.0
  cutoff for all trimesters.
