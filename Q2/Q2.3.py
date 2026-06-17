import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

FILE = Path(__file__).parent / "data.xlsx"
RF_ANNUAL = 0.0           # risk-free rate (annual).
ANN       = 12            # monthly -> annual scaling
NW_LAGS   = 4             # Newey-West (HAC) lags 
WIN       = 36            # rolling window length (months)

# %% [Load and clean] 
r = pd.read_excel(FILE, sheet_name="returns data")
r = r.sort_values("perf_date").reset_index(drop=True)        # one block was out of order

# Exclude Crowding (zero-padded pre-2008, insignificant) 
fac_cols = [c for c in r.columns if c.startswith("Factor") and c != "Factor - Crowding"]

# 6 corrupted cells (|monthly return| > 1 is impossible) -> NaN -> interpolate
r.loc[:, fac_cols] = r[fac_cols].mask(r[fac_cols].abs() > 1)
r[fac_cols] = r[fac_cols].interpolate(limit_direction="both")

print(f"{len(r)} months, {r.perf_date.min():%Y-%m} to {r.perf_date.max():%Y-%m}, "
      f"{len(fac_cols)} factors")

y = r["Hedge Fund"].values

# %% [Select the parsimonious model (AIC backward)]
keep = list(fac_cols)
cur = sm.OLS(y, sm.add_constant(r[keep])).fit()
while len(keep) > 1:
    aics = {c: sm.OLS(y, sm.add_constant(r[[k for k in keep if k != c]])).fit().aic for c in keep}
    drop = min(aics, key=aics.get)
    if aics[drop] < cur.aic:
        keep.remove(drop); cur = sm.OLS(y, sm.add_constant(r[keep])).fit()
    else:
        break
FINAL_FACTORS = keep
print("Parsimonious factors:", [k.replace("Factor - ", "") for k in FINAL_FACTORS])

# %% [The alpha test: parsimonious OLS + Newey-West]
# Decompose:  R_fund = alpha + beta . factors + e
X  = r[FINAL_FACTORS].values
m  = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})

betas = m.params[1:]
print(f"Alpha (annualised): {m.params[0]*ANN*100:6.2f}%   Newey-West t-stat: {m.tvalues[0]:.2f}")
print(f"R-squared:          {m.rsquared*100:6.1f}%   (share of fund variance the factors explain)")

loadings = pd.Series(betas, index=FINAL_FACTORS).sort_values(key=np.abs, ascending=False)
print("\nFactor loadings:")
print(loadings.round(2).to_string())

# %% [Sharpe ratios]
def perf(x, label):
    x   = np.asarray(x)
    mu  = x.mean() * ANN
    vol = x.std(ddof=1) * np.sqrt(ANN)
    sr  = (mu - RF_ANNUAL) / vol
    print(f"{label:28s} ret {mu*100:6.2f}%   vol {vol*100:6.2f}%   Sharpe {sr:5.2f}")
    return mu, vol, sr

repl_is = X @ betas          # in-sample replication: beta . factors (alpha stripped out)
print("Full sample:")
perf(y,       "Fund (as reported)")
perf(repl_is, "Replication (in-sample)")

# Estimate betas only on the trailing WIN months, then apply them to the NEXT month
oos_pred, oos_act, oos_date = [], [], []
for t in range(WIN, len(r)):
    bt = sm.OLS(y[t-WIN:t], sm.add_constant(X[t-WIN:t])).fit().params
    oos_pred.append(X[t] @ bt[1:])      # exclude alpha -> pure factor replica
    oos_act.append(y[t])
    oos_date.append(r["perf_date"].iloc[t])
oos_pred, oos_act = np.array(oos_pred), np.array(oos_act)

print(f"\nOut-of-sample ({len(oos_pred)} months, {WIN}m rolling window):")
perf(oos_act,  "Fund (same window)")
perf(oos_pred, "Replication (OOS)")

# %% [Stress test: is the alpha persistent or one crisis?]
# (a) rolling WIN-month annualised alpha + its t-stat
roll_a, roll_t, roll_d = [], [], []
for t in range(WIN, len(r)):
    f = sm.OLS(y[t-WIN:t], sm.add_constant(X[t-WIN:t])).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    roll_a.append(f.params[0] * ANN)
    roll_t.append(f.tvalues[0])
    roll_d.append(r["perf_date"].iloc[t])

# (b) sub-period alphas
bins   = pd.to_datetime(["2006-01-01", "2011-01-01", "2016-01-01", "2022-12-31"])
labels = ["2006-2010", "2011-2015", "2016-2022"]
r["_period"] = pd.cut(r["perf_date"], bins=bins, labels=labels, right=False)
print("\nSub-period alpha (annualised):")
for lab, g in r.groupby("_period", observed=True):
    f = sm.OLS(g["Hedge Fund"].values, sm.add_constant(g[FINAL_FACTORS].values)).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    print(f"  {lab}:  alpha {f.params[0]*ANN*100:6.2f}%   t {f.tvalues[0]:5.2f}   n={len(g)}")

# %% [Chart 1: cumulative growth of $1]
dates = r["perf_date"].values
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(dates, np.cumprod(1 + y),       lw=2.2, label="Fund (as reported)")
ax.plot(dates, np.cumprod(1 + repl_is), lw=1.8, label="Replication (in-sample)")
ax.plot(np.array(oos_date), np.cumprod(1 + oos_pred), lw=1.8, ls="--",
        label="Replication (out-of-sample)")
ax.axhline(1, color="grey", lw=0.6)
ax.set_title("Hedge fund vs. DIY factor replication — growth of $1")
ax.set_ylabel("Growth of $1")
ax.legend(frameon=False)
ax.grid(alpha=0.25)
plt.show()

# %% [Chart 2: rolling alpha — persistence check]
fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax1.plot(roll_d, np.array(roll_a) * 100, color="#16324f", label="rolling alpha (ann. %)")
ax1.axhline(0, color="grey", lw=0.6)
ax1.set_ylabel("annualised alpha (%)")
ax2 = ax1.twinx()
ax2.plot(roll_d, roll_t, color="#c8902a", alpha=0.75, label="t-stat")
ax2.axhline(2, color="#c8902a", ls=":", lw=1)
ax2.set_ylabel("Newey-West t-stat")
ax1.set_title(f"Rolling {WIN}-month alpha — is the edge persistent?")
fig.legend(loc="upper right", frameon=False)
plt.show()