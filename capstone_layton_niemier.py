"""
ISM2411 Capstone — Small-Business Loan Portfolio Analysis
Author: Layton Niemier
Date:   [Submission Date]

Business Questions:
  1. How do higher rates differ by credit score?
  2. Do higher rates even pay off over time because people that get higher
     loan rates tend to default on their loans?
  3. How does Construction's profit margin compare to Professional Services?

AI Disclosure: [List every AI tool you used and what you used it for]
"""

import numpy as np
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
# Question: How do higher rates differ by credit score?

# Only loans with a credit score can be plotted (blanks were left as NaN in CLEAN)
scored = df.dropna(subset=["credit_score"]).copy()

# Correlation: -1 = perfect downward relationship, 0 = none, +1 = perfect upward
correlation = scored["credit_score"].corr(scored["interest_rate"])
print(f"\nQ1 — Correlation between credit score and interest rate: {correlation:.2f}")

# Average rate within credit-score ranges, to have specific numbers to quote
credit_bins = [500, 600, 650, 700, 750, 850]
credit_labels = ["500-599", "600-649", "650-699", "700-749", "750+"]
scored["credit_band"] = pd.cut(scored["credit_score"], bins=credit_bins,
                               labels=credit_labels, right=False)
print("Average interest rate (%) by credit score range:")
print(scored.groupby("credit_band", observed=True)["interest_rate"].mean().round(2))

# Scatter plot: two number columns -> scatter is the right chart type.
# alpha makes overlapping dots see-through; the red line is the trend (best fit).
slope, intercept = np.polyfit(scored["credit_score"], scored["interest_rate"], 1)
x_line = np.linspace(scored["credit_score"].min(), scored["credit_score"].max(), 100)

plt.figure(figsize=(9, 5))
plt.scatter(scored["credit_score"], scored["interest_rate"], alpha=0.25, s=12)
plt.plot(x_line, slope * x_line + intercept, color="red", linewidth=2,
         label=f"Trend line (correlation = {correlation:.2f})")
plt.title("Interest Rate Charged vs. Borrower Credit Score")
plt.xlabel("Credit Score")
plt.ylabel("Interest Rate (%)")
plt.legend()
plt.tight_layout()
plt.savefig("charts/q1_rate_vs_credit_score.png", dpi=150, bbox_inches="tight")
plt.show()

# Finding: [Write your one-sentence finding here, with numbers]

# ── 6. BUSINESS QUESTION 2 ────────────────────────────────────
# Question: Do higher rates even pay off over time because people that get
#           higher loan rates tend to default on their loans?

# Group loans into interest-rate ranges
rate_bins = [6, 8, 10, 12, 14, 18]
rate_labels = ["6-7.99%", "8-9.99%", "10-11.99%", "12-13.99%", "14%+"]
df["rate_band"] = pd.cut(df["interest_rate"], bins=rate_bins,
                         labels=rate_labels, right=False)

# For each range: default rate (% of loans that defaulted) and
# profit margin (total profit / total lent, weighted by loan size like Q3)
rate_summary = df.groupby("rate_band", observed=True).agg(
    loans=("loan_id", "count"),
    total_lent=("loan_amount", "sum"),
    total_profit=("profit", "sum"),
    default_rate_pct=("loan_status", lambda s: (s == "Default").mean() * 100),
)
rate_summary["margin_pct"] = (
    rate_summary["total_profit"] / rate_summary["total_lent"] * 100
)
print("\nQ2 — Default rate and profit margin by interest-rate range:")
print(rate_summary[["loans", "default_rate_pct", "margin_pct"]].round(2))

# Two charts side by side in one image: left = defaults, right = profit margin
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(rate_summary.index.astype(str), rate_summary["default_rate_pct"],
        color="#d62728")
ax1.set_title("Default Rate by Interest Rate")
ax1.set_xlabel("Interest Rate Range")
ax1.set_ylabel("Loans Defaulted (%)")
ax1.tick_params(axis="x", rotation=30)

margin_colors = ["#2ca02c" if m >= 0 else "#d62728"
                 for m in rate_summary["margin_pct"]]
ax2.bar(rate_summary.index.astype(str), rate_summary["margin_pct"],
        color=margin_colors)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("Profit Margin by Interest Rate")
ax2.set_xlabel("Interest Rate Range")
ax2.set_ylabel("Profit Margin (% of Amount Lent)")
ax2.tick_params(axis="x", rotation=30)

fig.suptitle("Do Higher Interest Rates Pay Off?", fontsize=14)
plt.tight_layout()
plt.savefig("charts/q2_rates_defaults_margin.png", dpi=150, bbox_inches="tight")
plt.show()

# Finding: [Write your one-sentence finding here, with numbers]

# ── 7. BUSINESS QUESTION 3 ────────────────────────────────────
# Question: How does Construction's profit margin compare to Professional Services?
# All 8 industries are shown for context; the two being compared are highlighted.

# Portfolio profit margin per industry = total profit / total amount lent.
# Reason: this weights each loan by its size, the way an investor's actual
# return would be measured, instead of averaging small and large loans equally.
# Default rate and average interest rate are added to help explain WHY
# each industry's margin is high or low.
industry_margin = df.groupby("industry").agg(
    loans=("loan_id", "count"),
    total_profit=("profit", "sum"),
    total_lent=("loan_amount", "sum"),
    avg_rate_pct=("interest_rate", "mean"),
    default_rate_pct=("loan_status", lambda s: (s == "Default").mean() * 100),
)
industry_margin["margin_pct"] = (
    industry_margin["total_profit"] / industry_margin["total_lent"] * 100
)
industry_margin = industry_margin.sort_values("margin_pct", ascending=False)
print("\nQ3 — Profit margin, default rate and average rate by industry:")
print(industry_margin[["loans", "margin_pct", "default_rate_pct",
                       "avg_rate_pct"]].round(2))

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

# Finding: The industry with the higher margin was professional services.

# ── 8. SAVE CLEAN DATA ────────────────────────────────────────
df.to_csv("data/clean_data.csv", index=False)
print("\nCapstone complete. Clean data saved to data/clean_data.csv")
