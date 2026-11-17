"""
ISM2411 Capstone — Small-Business Loan Portfolio Analysis
Author: Layton Niemier
Date:   [Submission Date]

Business Questions:
  1. [Your question 1]
  2. [Your question 2]
  3. How does Construction's profit margin compare to Professional Services?

AI Disclosure: [List every AI tool you used and what you used it for]
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. LOAD ───────────────────────────────────────────────────
DATA_FILE = "data/capstone_finance.csv"
df = pd.read_csv(DATA_FILE)
print("Raw shape:", df.shape)
print(df.head())
df.info()
print(df.isnull().sum())

# Data quality issues found during inspection (M13):
#   - Column names have spaces and capitals ("Loan Amount", "Loan Status")
#   - Application Date: 30 blanks, mixed formats (2025-01-01, 01/10/2025,
#     January 30 2025), and invalid entries (00/00/0000, 2025-02-30, TBD, pending)
#   - Loan Amount and Profit stored as text with "$" and ","
#   - Profit has negative values (losses on defaulted loans), which are real
#   - Industry has typos ("Resturants", "constrction"), mixed case, extra spaces
#   - Region: 128 blanks
#   - Credit Score: 91 blanks
#   - Employees: 9999 appears as a placeholder (describe() max)
#   - 40 exact duplicate rows

# ── 2. CLEAN ──────────────────────────────────────────────────
# Document every decision with a comment explaining WHY.

# Standardize column names
# Reason: lowercase snake_case names are easier to type and less error-prone
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Remove exact duplicates first
# Reason: a loan entered twice would double-count its amount and profit
df = df.drop_duplicates()

# Parse application_date — the file mixes several formats, so use format="mixed";
# invalid entries (00/00/0000, 2025-02-30, TBD, pending, unknown) become NaT
df["application_date"] = pd.to_datetime(
    df["application_date"], format="mixed", errors="coerce"
)

# Drop rows with no usable application date
# Reason: we cannot place a loan in a month or quarter without its date
df = df.dropna(subset=["application_date"])

# Fix money columns: strip "$" and ",", convert to float
# Reason: amounts stored as text cannot be summed or averaged
for col in ["loan_amount", "profit"]:
    df[col] = (
        df[col].str.replace("$", "", regex=False)
               .str.replace(",", "", regex=False)
               .astype(float)
    )

# Keep negative profit values
# Reason: a negative profit is the lender's loss on a defaulted loan, which is
# real and important, not a data error

# Remove impossible employee counts
# Reason: 9999 is a placeholder someone typed, not a real small-business headcount
df = df[df["employees"] != 9999]

# Fix industry typos: strip spaces and fix capitalization, then correct misspellings
# Reason: "retail", "RETAIL" and "Retail " are the same industry and must group together
df["industry"] = df["industry"].str.strip().str.title()
df["industry"] = df["industry"].replace(
    {"Resturants": "Restaurants", "Constrction": "Construction"}
)

# Fill missing region with "Unknown"
# Reason: the loan amount and profit are still valid, so keep the row
# instead of losing it
df["region"] = df["region"].fillna("Unknown")

# Leave missing credit_score as NaN (not filled)
# Reason: inventing a credit score with the median could distort any
# risk analysis; pandas skips NaN in means and groupbys

print("Clean shape:", df.shape)
print(df["industry"].value_counts())

# ── 3. DERIVED COLUMNS ────────────────────────────────────────
df["month"] = df["application_date"].dt.month
df["quarter"] = df["application_date"].dt.quarter
# Profit as a % of the amount lent, so loans of different sizes can be compared
df["profit_margin_pct"] = df["profit"] / df["loan_amount"] * 100

# ── 4. DESCRIPTIVE SUMMARY ───────────────────────────────────
print("\nDescriptive stats (clean):")
print(df.describe())
print("\nPortfolio summary by loan status:")
print(
    df.groupby("loan_status").agg(
        loans=("loan_id", "count"),
        total_lent=("loan_amount", "sum"),
        total_profit=("profit", "sum"),
    ).sort_values("total_lent", ascending=False)
)

# ── 5. BUSINESS QUESTION 1 ────────────────────────────────────
# [State your question in a comment]
# [Your analysis code here]
# [Your chart code here — with title, axis labels, savefig, show]
# [Write a one-sentence finding in a comment]

# ── 6. BUSINESS QUESTION 2 ────────────────────────────────────
# [State your question]
# [Analysis + chart]

# ── 7. BUSINESS QUESTION 3 ────────────────────────────────────
# Question: How does Construction's profit margin compare to Professional Services?
# All 8 industries are shown for context; the two being compared are highlighted.

# Portfolio profit margin per industry = total profit / total amount lent.
# Reason: this weights each loan by its size, the way an investor's actual
# return would be measured, instead of averaging small and large loans equally.
industry_margin = df.groupby("industry").agg(
    total_profit=("profit", "sum"),
    total_lent=("loan_amount", "sum"),
)
industry_margin["margin_pct"] = (
    industry_margin["total_profit"] / industry_margin["total_lent"] * 100
)
industry_margin = industry_margin.sort_values("margin_pct", ascending=False)
print("\nQ3 — Profit margin by industry (%):")
print(industry_margin["margin_pct"].round(2))

highlight = ["Construction", "Professional Services"]
colors = ["#1f77b4" if name in highlight else "#c7c7c7"
          for name in industry_margin.index]

plt.figure(figsize=(9, 5))
plt.bar(industry_margin.index, industry_margin["margin_pct"], color=colors)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Portfolio Profit Margin by Industry\n"
          "(Construction and Professional Services highlighted)")
plt.xlabel("Industry")
plt.ylabel("Profit Margin (% of Amount Lent)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("charts/q3_margin_by_industry.png", dpi=150, bbox_inches="tight")
plt.show()

# Finding: [Write your one-sentence finding here, with the numbers for
#           Construction and Professional Services]

# ── 8. SAVE CLEAN DATA ────────────────────────────────────────
df.to_csv("data/clean_data.csv", index=False)
print("\nCapstone complete. Clean data saved to data/clean_data.csv")
