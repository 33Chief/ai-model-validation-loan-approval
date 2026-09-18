# AI Model Validation Report — Consumer Loan Approval Model

An independent, end-to-end **model risk validation** of a synthetic consumer loan approval model — performance testing, stability/drift testing, and a fairness (disparate impact) audit, written up as a formal validation report the way a bank Model Risk Management (MRM) or AI Governance team would deliver it.

**[Read the full Model Validation Report (PDF)](outputs/Model_Validation_Report.pdf)**

## Why this project

Most "ML portfolio projects" stop at model accuracy. This one goes one step further and asks the question a risk/compliance function actually gets paid to ask: **is this model safe, stable, and fair enough to put into production — and if not, exactly what needs to change?**

## What's inside

- **`src/01_generate_data.py`** — generates a synthetic loan-application dataset. Applicant creditworthiness is generated independently of protected class, but a "neighborhood risk tier" feature is deliberately correlated with protected group membership — simulating a realistic **proxy discrimination** scenario, the kind fair-lending reviews are designed to catch.
- **`src/02_train_model.py`** — trains a gradient-boosted classifier. Protected attributes (gender, protected group) are excluded from the feature set, consistent with fair-lending practice (ECOA / Reg B), and held out separately for independent testing.
- **`src/03_validate.py`** — the validation suite:
  - **Performance:** accuracy, precision/recall/F1, ROC-AUC, calibration, confusion matrix
  - **Stability:** Population Stability Index (PSI) against a simulated forward/stress period
  - **Fairness:** demographic parity, equalized odds, and the four-fifths (80%) rule, via [Fairlearn](https://fairlearn.org/), across protected group and gender
  - **Data quality:** completeness, duplication, class balance, outlier scan
- **`src/04_build_report.js`** — compiles all of the above into a formal Word/PDF validation report, including a section mapping every activity to the **NIST AI Risk Management Framework** (Govern / Map / Measure / Manage).

## Key finding

The audit surfaces a real, material issue: the model's `zip_tier` feature acts as a proxy for protected group membership, producing a demographic parity ratio of **0.801** — technically clearing the common 80% four-fifths screening threshold, but by a margin of only 0.001, corroborated by a 9.2-point equalized-odds gap. The report traces the mechanism, rates it **Material**, and lays out a remediation and monitoring plan — rather than stopping at "the model passed."

## Stack

Python (scikit-learn, pandas, Fairlearn, matplotlib) for the modeling and validation suite · Node.js (`docx`) for the report · LibreOffice/Poppler for PDF rendering.

## Reproduce it

```bash
pip install pandas numpy scikit-learn matplotlib fairlearn joblib
python src/01_generate_data.py
python src/02_train_model.py
python src/03_validate.py
node src/04_build_report.js
```

---
*All applicant data is synthetically generated for demonstration purposes and does not represent real individuals.*
