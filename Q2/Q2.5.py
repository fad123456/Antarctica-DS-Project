import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from scipy.stats import f as f_dist
import matplotlib.pyplot as plt
from pathlib import Path

FILE = Path(__file__).parent / "data.xlsx"
ANN     = 12             # monthly -> annual
NW_LAGS = 4              # Newey-West lags 
WIN     = 36             # rolling window length (months)

# %% [Load and clean] 
r = pd.read_excel(FILE, sheet_name="returns data")
r = r.sort_values("perf_date").reset_index(drop=True)

# Exclude Crowding (zero-padded pre-2008, insignificant) 
fac_all = [c for c in r.columns if c.startswith("Factor") and c != "Factor - Crowding"]
r.loc[:, fac_all] = r[fac_all].mask(r[fac_all].abs() > 1)   # kill 6 corrupted cells
r[fac_all] = r[fac_all].interpolate(limit_direction="both")

y = r["Hedge Fund"].values

# %% [Select the parsimonious model (AIC backward)]
keep = list(fac_all)
cur = sm.OLS(y, sm.add_constant(r[keep])).fit()
while len(keep) > 1:
    aics = {c: sm.OLS(y, sm.add_constant(r[[k for k in keep if k != c]])).fit().aic for c in keep}
    drop = min(aics, key=aics.get)
    if aics[drop] < cur.aic:
        keep.remove(drop); cur = sm.OLS(y, sm.add_constant(r[keep])).fit()
    else:
        break
fac_cols = keep      # the betas we test for stationarity: Low Risk, Value/Growth, EM, Credit
print("Testing stationarity of parsimonious betas:", [c.replace("Factor - ", "") for c in fac_cols])

X  = r[fac_cols].values
Xc = sm.add_constant(X)

# %% [Static (full-sample) betas - the baseline we're testing]
full       = sm.OLS(y, Xc).fit()
betas_full = pd.Series(full.params[1:], index=fac_cols)

# %% [Rolling betas + instability summary] 
roll = []
for t in range(WIN, len(r)):
    bt = sm.OLS(y[t-WIN:t], sm.add_constant(X[t-WIN:t])).fit().params[1:]
    roll.append(bt)
roll = pd.DataFrame(roll, columns=fac_cols, index=r["perf_date"].iloc[WIN:].values)

summary = pd.DataFrame({
    "full_beta":  betas_full,
    "roll_min":   roll.min(),
    "roll_max":   roll.max(),
    "roll_sd":    roll.std(),
    "sign_flips": (np.sign(roll).diff().fillna(0) != 0).sum(),
}).reindex(betas_full.abs().sort_values(ascending=False).index)

print("\nRolling-beta instability (sorted by |full-sample beta|):")
print(summary.round(2).to_string())

# %% [OLS-CUSUM: omnibus parameter-stability test] 
# H0 = coefficients are stable. Low power vs slope/offsetting breaks, so read it
# alongside the Chow tests rather than on its own.
stat, p, crit = breaks_cusumolsresid(full.resid, ddof=Xc.shape[1])
print(f"\nOLS-CUSUM (H0 = stable betas):  stat={stat:.3f}  p-value={p:.4f}")

# %% [Chow tests: targeted structural breaks at suspected regime dates]
def chow_test(breakpoint):
    k, n = Xc.shape[1], len(y)
    ssr = lambda a, b: sm.OLS(y[a:b], Xc[a:b]).fit().ssr
    s1, s2 = ssr(0, breakpoint), ssr(breakpoint, n)
    F = ((full.ssr - (s1 + s2)) / k) / ((s1 + s2) / (n - 2 * k))
    return F, 1 - f_dist.cdf(F, k, n - 2 * k)

print("\nChow structural-break tests (H0 = no break):")
for label, date in [("end-2007 (pre-GFC)",   "2008-01-31"),
                    ("end-2010 (alpha shift)","2011-01-31"),
                    ("end-2015",              "2016-01-31")]:
    bp = int((r["perf_date"] < date).sum())
    F, pv = chow_test(bp)
    flag = "  <-- significant" if pv < 0.05 else ""
    print(f"  {label:24s} bp={bp:3d}:  F={F:5.2f}  p={pv:.4f}{flag}")

# %% [Chart: rolling betas vs their static value] 
sel    = betas_full.abs().sort_values(ascending=False).index[:3]
colors = ["#16324f", "#c8902a", "#9b2226"]
fig, ax = plt.subplots(figsize=(10, 5))
for f, c in zip(sel, colors):
    ax.plot(roll.index, roll[f], color=c, lw=1.7, label=f.replace("Factor - ", ""))
    ax.axhline(betas_full[f], color=c, ls=":", lw=1, alpha=0.7)
ax.axhline(0, color="grey", lw=0.6)
ax.set_title("Rolling 36m factor betas (dotted = static full-sample beta)")
ax.set_ylabel("beta")
ax.legend(frameon=False, title="rolling beta")
ax.grid(alpha=0.25)
plt.show()