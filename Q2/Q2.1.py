import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from pathlib import Path

FILE       = Path(__file__).parent / "data.xlsx"
DATE_COL   = "perf_date"
FUND_COL   = "Hedge Fund"
DROP_COLS  = ["Factor - Crowding"]   # excluded: pre-2008 inception + insignificant
ABS_THRESH = 1.0                     # |monthly return| > 100% flags a corrupted cell
HAC_LAGS   = 4                       # Newey-West; ~ 4*(T/100)^(2/9) for T~195


df = pd.read_excel(FILE, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
df = df.drop(columns=DROP_COLS)
factor_cols = [c for c in df.columns if c not in (DATE_COL, FUND_COL)]

# clean corrupted cells 
flagged = {}
for c in [FUND_COL] + factor_cols:
    mask = df[c].abs() > ABS_THRESH
    if mask.any():
        flagged[c] = list(df.loc[mask, DATE_COL].dt.date.astype(str))
        df.loc[mask, c] = np.nan
        df[c] = df[c].interpolate(method="linear", limit_direction="both")
print("Corrupted cells repaired (interpolated):", flagged)

model_df = df.dropna(subset=[FUND_COL] + factor_cols).reset_index(drop=True)
print(f"Sample: {len(model_df)} months, "
      f"{model_df[DATE_COL].min().date()} -> {model_df[DATE_COL].max().date()}")


y = model_df[FUND_COL]
X = sm.add_constant(model_df[factor_cols])
ols = sm.OLS(y, X).fit()
hac = ols.get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS)

res = pd.DataFrame({"coef": hac.params, "HAC_se": hac.bse,
                    "t": hac.tvalues, "p": hac.pvalues},
                   index=["alpha (const)"] + factor_cols)
print("\nFull model (HAC, lag 4)")
print(res.round(4).to_string())

a = hac.params[0]
print(f"\nalpha monthly        = {a:.5f}")
print(f"alpha annual (x12)   = {a*12:.5f}")
print(f"alpha annual (geom.) = {(1+a)**12 - 1:.5f}")
print(f"R^2 = {ols.rsquared:.4f}   adj R^2 = {ols.rsquared_adj:.4f}")


vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns).drop("const")
print("\nVIF (max = %.2f)" % vif.max())
print(vif.round(2).to_string())

bp = het_breuschpagan(ols.resid, ols.model.exog)
jb = jarque_bera(ols.resid)
print(f"\nDurbin-Watson      = {durbin_watson(ols.resid):.3f}")
print(f"Breusch-Pagan LM p = {bp[1]:.4f}")
print(f"Jarque-Bera p      = {jb[1]:.4f} (skew={jb[2]:.2f}, kurt={jb[3]:.2f})")

keep = list(factor_cols)
cur = sm.OLS(y, sm.add_constant(model_df[keep])).fit()
improved = True
while improved and len(keep) > 1:
    improved = False
    trials = {c: sm.OLS(y, sm.add_constant(model_df[[k for k in keep if k != c]])).fit().aic
              for c in keep}
    drop, best = min(trials.items(), key=lambda kv: kv[1])
    if best < cur.aic:
        keep.remove(drop)
        cur = sm.OLS(y, sm.add_constant(model_df[keep])).fit()
        improved = True

print("\nParsimonious (AIC backward)")
print("retained:", keep)
print(f"alpha monthly = {cur.params['const']:.5f}  annual(x12) = {cur.params['const']*12:.5f}")
print(f"adj R^2 = {cur.rsquared_adj:.4f}   (full adj R^2 = {ols.rsquared_adj:.4f})")