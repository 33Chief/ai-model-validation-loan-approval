"""
Generate a synthetic small-business/consumer loan approval dataset.

Design intent (for the validation report to be meaningful):
- Applicant *creditworthiness* (credit_score, income, DTI, delinquencies) is
  generated INDEPENDENTLY of protected_group and gender -> in the real world,
  these groups are equally creditworthy on average.
- protected_group is correlated with zip_tier (a "neighborhood risk tier"
  feature a lender might legitimately want to use for regional pricing).
- The approval label is generated from legitimate credit factors PLUS a small
  negative weight on zip_tier. Because zip_tier proxies for protected_group,
  this creates realistic disparate impact WITHOUT the model ever seeing
  gender or protected_group directly - the classic "proxy discrimination"
  pattern model risk / fair lending teams are trained to catch.
- gender also gets a small direct penalty, independent of zip_tier, so the
  report has two distinct findings of different severity/mechanism.

protected_group and gender are NOT used as model features (excluded per
fair-lending practice / ECOA) but are retained in the held-out test set
so the validation team can run a fairness audit against them, mirroring how
a real MRM team receives a "protected class" file from compliance to test
against, separate from the production feature set.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 6000

# --- Protected attributes (not used as model features) ---
protected_group = rng.choice(["Group A", "Group B"], size=N, p=[0.68, 0.32])
gender = rng.choice(["Male", "Female"], size=N, p=[0.52, 0.48])

# --- zip_tier: legitimate-looking feature, correlated with protected_group ---
# Tier 1 = lowest regional risk, Tier 5 = highest. Group B applicants are
# disproportionately concentrated in higher tiers (proxy correlation).
zip_tier = np.where(
    protected_group == "Group A",
    rng.choice([1, 2, 3, 4, 5], size=N, p=[0.28, 0.27, 0.22, 0.14, 0.09]),
    rng.choice([1, 2, 3, 4, 5], size=N, p=[0.10, 0.16, 0.22, 0.27, 0.25]),
)

# --- True creditworthiness: independent of group/gender ---
credit_score = np.clip(rng.normal(680, 65, N), 300, 850).round().astype(int)
annual_income = np.clip(rng.normal(68000, 24000, N), 18000, 260000).round(-2)
debt_to_income = np.clip(rng.normal(0.34, 0.11, N), 0.02, 0.85).round(3)
employment_length = np.clip(rng.exponential(5, N), 0, 35).round(1)
age = np.clip(rng.normal(41, 12, N), 21, 75).round().astype(int)
num_open_accounts = np.clip(rng.poisson(6, N), 0, 25)
delinquencies_2yr = np.clip(rng.poisson(0.35, N), 0, 8)
loan_amount = np.clip(rng.normal(18000, 9000, N), 1000, 75000).round(-2)

# --- Approval score: legitimate factors + small proxy bias + small direct gender bias ---
z = (
    0.011 * (credit_score - 680)
    + 0.000014 * (annual_income - 68000)
    - 2.6 * (debt_to_income - 0.34)
    + 0.05 * employment_length
    - 0.22 * delinquencies_2yr
    - 0.045 * num_open_accounts
    - 0.00003 * (loan_amount - 18000)
    - 0.16 * (zip_tier - 3)          # proxy penalty, rises with zip_tier
    - 0.14 * (gender == "Female")     # small direct gender penalty
    + rng.normal(0, 0.55, N)          # noise
)
prob_approve = 1 / (1 + np.exp(-z))
loan_approved = (rng.uniform(0, 1, N) < prob_approve).astype(int)

df = pd.DataFrame({
    "applicant_id": [f"APP{100000+i}" for i in range(N)],
    "credit_score": credit_score,
    "annual_income": annual_income,
    "debt_to_income": debt_to_income,
    "employment_length_years": employment_length,
    "age": age,
    "num_open_accounts": num_open_accounts,
    "delinquencies_2yr": delinquencies_2yr,
    "loan_amount_requested": loan_amount,
    "zip_tier": zip_tier,
    "protected_group": protected_group,   # held out - not a model feature
    "gender": gender,                     # held out - not a model feature
    "loan_approved": loan_approved,
})

# Inject a small amount of realistic missingness for the data-quality section
miss_idx = rng.choice(N, size=int(N * 0.012), replace=False)
df.loc[miss_idx, "employment_length_years"] = np.nan

df.to_csv("/home/claude/ai_model_validation_project/data/loan_applications.csv", index=False)
print(df.shape)
print(df["loan_approved"].mean())
print(df.groupby("protected_group")["loan_approved"].mean())
print(df.groupby("gender")["loan_approved"].mean())
