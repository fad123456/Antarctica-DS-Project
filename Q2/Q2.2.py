import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
from pathlib import Path

FILE = Path(__file__).parent / "data.xlsx"
DATE_COL   = "perf_date"
FUND_COL   = "Hedge Fund"
CROWD_COL  = "Factor - Crowding"
MARKET_COL = "Factor - Local Equity"
ABS_THRESH = 1.0
HAC_LAGS   = 4
WINDOW     = 36


def calendar_span(dates):
    a, b = dates.iloc[0], dates.iloc[-1]
    return (b.year - a.year) * 12 + (b.month - a.month) + 1

def fit_hac(y, X):
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}, use_t=True)


# Cleaned spine
df = pd.read_excel(FILE, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

# Exclude Crowding: zero-padded before its 2008 inception and statistically insignificant.
# Dropping the COLUMN (not the rows) keeps the full 2006-2022 sample for the other factors.
df = df.drop(columns=[CROWD_COL])
factor_cols = [c for c in df.columns if c not in (DATE_COL, FUND_COL)]

# Repair the six corrupted cells (|monthly return| > 100%) by linear interpolation, preserving an unbroken monthly series so rolling windows have no calendar gaps.
n_out = {}
for c in [FUND_COL] + factor_cols:
    mask = df[c].abs() > ABS_THRESH
    n_out[c] = int(mask.sum())
    df.loc[mask, c] = np.nan
    df[c] = df[c].interpolate(method="linear", limit_direction="both")
print("Corrupted cells repaired (interpolated):", {k: v for k, v in n_out.items() if v})

model_df = df.dropna(subset=[FUND_COL] + factor_cols).reset_index(drop=True)
d = model_df[DATE_COL]
gaps = [(d.iloc[i-1].date(), d.iloc[i].date(),
         (d.iloc[i].year-d.iloc[i-1].year)*12 + (d.iloc[i].month-d.iloc[i-1].month))
        for i in range(1, len(d))
        if (d.iloc[i].year-d.iloc[i-1].year)*12 + (d.iloc[i].month-d.iloc[i-1].month) != 1]
print(f"Sample: {len(model_df)} rows, {d.min().date()} -> {d.max().date()}")
print("Interior calendar gaps:", gaps if gaps else "none")

# Full model: OLS coefficients, HAC standard errors
y = model_df[FUND_COL].astype(float)
X_full = sm.add_constant(model_df[factor_cols].astype(float))
full = fit_hac(y, X_full)
res = pd.DataFrame({"coef": full.params, "HAC_se": full.bse, "t": full.tvalues, "p": full.pvalues})
print("\n=== Full model: alpha & betas (HAC, lag 4) ===")
print(res.round(4).to_string())
a = full.params["const"]
print(f"\nalpha monthly        = {a:.5f}")
print(f"alpha annual (x12)   = {a*12:.5f}")
print(f"alpha annual (geom.) = {(1+a)**12 - 1:.5f}")
print(f"R^2 = {full.rsquared:.4f}   adj R^2 = {full.rsquared_adj:.4f}")

# Diagnostics: VIF, Durbin-Watson, Breusch-Pagan, Jarque-Bera
vif = pd.Series([variance_inflation_factor(X_full.values, i)
                 for i in range(X_full.shape[1])], index=X_full.columns).drop("const")
print("\n=== VIF (flag >5, serious >10) ===")
print(vif.round(2).to_string()); print("max VIF =", round(vif.max(), 2))
bp = het_breuschpagan(full.resid, full.model.exog)
jb = jarque_bera(full.resid)
print(f"\nDurbin-Watson = {durbin_watson(full.resid):.3f}")
print(f"Breusch-Pagan p = {bp[1]:.4f}")
print(f"Jarque-Bera p = {jb[1]:.4f} (skew={jb[2]:.2f}, kurt={jb[3]:.2f})")

# AIC backward elimination 
keep = list(factor_cols)
cur = sm.OLS(y, sm.add_constant(model_df[keep])).fit()
while len(keep) > 1:
    aics = {c: sm.OLS(y, sm.add_constant(model_df[[k for k in keep if k != c]])).fit().aic for c in keep}
    drop = min(aics, key=aics.get)
    if aics[drop] < cur.aic:
        keep.remove(drop); cur = sm.OLS(y, sm.add_constant(model_df[keep])).fit()
    else:
        break
FINAL_FACTORS = keep
X_f = sm.add_constant(model_df[FINAL_FACTORS].astype(float), has_constant="add")
final = fit_hac(y, X_f)
a_f = final.params["const"]
print("\n=== Parsimonious model (AIC backward) ===")
print("retained:", FINAL_FACTORS)
print(f"alpha monthly = {a_f:.5f}   annual(x12) = {a_f*12:.5f}")
print(f"adj R^2 = {final.rsquared_adj:.4f}   (full adj R^2 = {full.rsquared_adj:.4f})")
print("alpha is", "STABLE" if abs(a_f - a) < 0.002 else "NOT stable", "across specifications")

# Rolling alpha and betas (36-month trailing windows) 
rows = []
for end in range(WINDOW, len(model_df) + 1):
    s = model_df.iloc[end - WINDOW:end]
    m = fit_hac(s[FUND_COL], sm.add_constant(s[FINAL_FACTORS], has_constant="add"))
    lo, hi = m.conf_int().loc["const"]
    rec = {"date": s[DATE_COL].iloc[-1], "annual_alpha": 12*m.params["const"],
           "lo": 12*lo, "hi": 12*hi, "span": calendar_span(s[DATE_COL])}
    for fac in FINAL_FACTORS:
        rec[fac] = m.params[fac]
    rows.append(rec)
roll = pd.DataFrame(rows)
straddle = roll[roll["span"] > WINDOW]
print(f"\nRolling alpha: mean {roll.annual_alpha.mean():.3f}, "
      f"min {roll.annual_alpha.min():.3f}, max {roll.annual_alpha.max():.3f}, "
      f"share>0 {(roll.annual_alpha>0).mean():.0%}")

# cumulative contribution by source (alpha, betas, residual)
betas = final.params.drop("const")
contrib = model_df[FINAL_FACTORS].multiply(betas, axis=1)
contrib["Alpha"] = a_f
contrib["Residual (unexplained)"] = final.resid.values
comps = ["Alpha"] + FINAL_FACTORS + ["Residual (unexplained)"]
cum = contrib[comps].cumsum(); cum.index = model_df[DATE_COL]
print("\nCumulative contribution by source:")
print(contrib[comps].sum().sort_values().round(3).to_string())

# Regime analysis (high vs normal volatility) 
vol = model_df[MARKET_COL].rolling(12).std()
thr = vol.quantile(0.75)
model_df["regime"] = (pd.Series(np.where(vol > thr, "High vol", "Normal vol"),
                                index=model_df.index).where(vol.notna()))
reg = model_df.dropna(subset=["regime"]).reset_index(drop=True)
desc = []
for name, s in reg.groupby("regime"):
    r = sm.OLS(s[FUND_COL], sm.add_constant(s[FINAL_FACTORS], has_constant="add")).fit()
    desc.append({"regime": name, "n": len(s), "annual_alpha": 12*r.params["const"], "R2": round(r.rsquared, 3)})
print("\nRegime alpha (descriptive point estimates):")
print(pd.DataFrame(desc).round(4).to_string(index=False))
D = (reg["regime"] == "High vol").astype(float)
Xi = pd.DataFrame({"const": 1.0, "high_vol": D})
for fac in FINAL_FACTORS:
    Xi[fac] = reg[fac].values
    Xi[fac + " x high_vol"] = reg[fac].values * D.values
inter = fit_hac(reg[FUND_COL], Xi)
print("\nRegime alpha (valid interaction test):")
print(f"  Normal-vol alpha (x12) = {12*inter.params['const']:.4f}")
print(f"  High-vol alpha   (x12) = {12*(inter.params['const']+inter.params['high_vol']):.4f}")
print(f"  Difference             = {12*inter.params['high_vol']:.4f} (p = {inter.pvalues['high_vol']:.4f})")

#charts
fig1, ax = plt.subplots(figsize=(11, 5))
ax.plot(roll["date"], roll["annual_alpha"], color="#1f3a5f", lw=1.8, label="36m rolling alpha (ann.)")
ax.fill_between(roll["date"], roll["lo"], roll["hi"], color="#1f3a5f", alpha=0.15, label="95% CI")
ax.axhline(0, color="grey", lw=1)
ax.axhline(a_f*12, color="#c0392b", ls="--", lw=1.2, label=f"full-sample alpha ({a_f*12:.1%})")
if len(straddle):
    ax.axvspan(straddle["date"].min(), straddle["date"].max(), color="orange", alpha=0.12, label="gap-straddling windows")
ax.set(title="36-Month Rolling Fund Alpha", xlabel="Window end date", ylabel="Annualised alpha")
ax.legend(loc="upper left", fontsize=8); fig1.tight_layout()

fig2, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for ax, fac in zip(axes.ravel(), FINAL_FACTORS):
    ax.plot(roll["date"], roll[fac], color="#1f3a5f", lw=1.6)
    ax.axhline(final.params[fac], color="#c0392b", ls="--", lw=1.1, label=f"full-sample β = {final.params[fac]:.2f}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(fac, fontsize=10); ax.legend(fontsize=8)
fig2.suptitle("36-Month Rolling Factor Betas"); fig2.tight_layout()

fig3, ax = plt.subplots(figsize=(12, 6))
for c in comps:
    lw = 2.2 if c in ("Alpha", "Residual (unexplained)") else 1.3
    ax.plot(cum.index, cum[c], label=c, lw=lw)
ax.plot(model_df[DATE_COL], model_df[FUND_COL].cumsum(), color="black", lw=2.2, ls="--", label="Actual fund (cumulative)")
ax.axhline(0, color="grey", lw=1)
ax.set(title="Cumulative Attribution incl. Residual", xlabel="Date", ylabel="Cumulative additive return")
ax.legend(fontsize=8, ncol=2); fig3.tight_layout()

plt.show()