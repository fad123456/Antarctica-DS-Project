# AI Declaration
I have used AI in every question for the following purposes:
- Research and Brainstorming
- Debugging
- Generating scripts and specific functions to illustrate key ideas

# Q1 - GitHub repository lister (no GitHub API / no PyGithub).

Reads a GitHub username and a destination folder from a small GUI.
When the button is pressed, it scrapes the user's public "Repositories"
web page (plain HTML, not the API) and writes an Excel file listing
every repository name to the chosen folder.

Methodology: 1. Request each repository page in sequence. 2. Parse the returned HTML with BeautifulSoup. 3. Extract repository names using GitHub's repository-link attribute. 4. Stop when a page contains no repository links.

# Q2 - Multilinear Regression analysis
# 2.1.

I modelled the fund's monthly return as a linear function of the different investment factors 

R_fund,t = α + Σ βᵢ · Factorᵢ,t + εₜ

To clean the data beforehand I:
- excluded the crowding factor because it had no meaningful observations before 2008 and did not improve the model. 
- replaced six implausibly large factor returns using linear interpolation. 
- sorted the data in chronological order.

The initial model included all remaining factors and was estimated using ordinary least squares. Newey–West HAC standard errors with four lags were used to account for possible autocorrelation and heteroskedasticity.

The full model produced a monthly alpha of 0.00921, equivalent to 0.921% per month or approximately 11.05% per year using simple annualisation. The alpha was statistically significant with a HAC p-value of approximately 0.0001.

The full model had an (R^2) of 0.3550 and an adjusted (R^2) of 0.2931. The strongest statistically significant factor exposures were:

Value vs Growth beta: −0.65
Credit beta: 0.23

Variance inflation factors were all below 3.30, indicating that multicollinearity was not a serious problem. The Breusch–Pagan test found no evidence of heteroskedasticity. The Jarque–Bera test indicated some non-normality in the residuals, while the Durbin–Watson statistic suggested mild positive serial correlation.

An AIC backward selection retains four factors (Low Risk, Value vs Growth, Emerging Markets, Credit), raises adj R² to 0.31, and leaves alpha materially unchanged (0.0076/month, 9.16% annualised). This confirms the result is carried by a few factors and is not an artefact of overfitting. I keep the full model as the primary specification for reporting alpha and betas, because stepwise selection invalidates the standard errors of the retained coefficients (post-selection inference), and treat the AIC model only as confirmation.


# 2.2.

I evaluate the model on three dimensions: how much it explains, whether its estimates are statistically valid, and whether they are stable over time.

The full model explains 36% of the variation in monthly fund returns, with an adjusted (R^2) of 0.29. The estimated alpha is positive and highly significant at 0.00921 per month ((t=4.04), (p<0.001)). The clearest factor exposures are Value vs Growth (beta=-0.65), (t=-7.00) and Credit (beta=0.23), (t=2.50), while Emerging Markets is only weakly significant ((p=0.087)).

The parsimonious model gives a similar result, with alpha of 0.0076 per month, or approximately 9.2% annualised, and a slightly higher adjusted (R^2) of 0.31. This suggests that the positive alpha is not simply caused by overfitting.

The maximum VIF is 3.30, indicating no serious multicollinearity as we saw previously. The Breusch–Pagan test ((p=0.99)) finds no evidence of heteroskedasticity, and the Ljung–Box test ((p=0.48)) finds no meaningful autocorrelation. Newey–West HAC standard errors are reported as a precaution and do not change the conclusions. The Jarque–Bera test ((p=0.032)) indicates mild non-normality, but this is less concerning given the sample size and robust standard errors.

The main limitation is parameter instability. Rolling 36-month regressions show that annualised alpha varies from slightly negative to around 18%, although it remains positive in 92% of windows. Several factor betas also change materially over time, with some changing sign. The full-sample coefficients should therefore be interpreted as average exposures rather than fixed relationships.

The cumulative-return decomposition shows that alpha is the dominant contributor, while individual factor contributions are relatively small. This is consistent with manager-specific performance, although it should not be treated as definitive proof of skill.

Finally, alpha is higher in high-volatility periods than in normal periods, but the difference is not statistically significant ((p=0.63)). Overall, the model provides a useful average description of the fund, but the instability of alpha and betas is an important limitation for the risk analysis in Section 2.5.

# 2.4.

From the figures in 2.3., the fund's annualised volatility (10.3%) is nearly double the factor portfolio's (5.9%). This follows from the regression, since alpha is constant, Var(fund) = Var(β·factors) + Var(residual). The factor portfolio captures only the first term, which is 33% of the fund's variance (the R²); the other 67% is idiosyncratic residual risk. So the fund's extra volatility is precisely the fund-specific risk the factors cannot reproduce, and the factor portfolio's volatility is √0.33 = 0.57 times the fund's.
However, the factor portfolio is riskier in the tails. Despite lower volatility, it is far more negatively skewed (−0.99 vs −0.25) and fatter-tailed (kurtosis 6.6 vs 3.6), and its worst drawdown (−21.6%) is almost as deep as the fund's (−23.4%) at only 57% of the volatility. Its risk is concentrated in crashes, because the systematic factor exposures move together in stress periods.

Thus, by total volatility the fund is more risky, and that extra risk is idiosyncratic. But it is the more risk-efficient strategy: its Sharpe (0.99 vs 0.16) shows the higher volatility is well rewarded, whereas the factor portfolio bears worse tail risk for almost no return.

# 2.5. 

To check whether the fund's factor exposures hold steady over time, I re-estimated the four parsimonious betas (Low Risk, Value/Growth, Emerging Markets, and Credit) over rolling 36-month windows and backed this up with two formal tests: an OLS-CUSUM test for overall parameter stability and Chow tests for breaks at specific dates.

The rolling estimates show that the betas do not hold still. Each one drifts across a wide range and repeatedly changes sign, with a rolling standard deviation about as large as the beta itself. The Value/Growth sits near zero before 2012 but falls to roughly −1.0 after 2018, so the single full-sample figure of −0.62 does not really describe any period well. The Emerging Markets beta is even more restless, switching sign seventeen times over the sample. In short, the static betas are averages of exposures that were constantly moving, not stable constants.

The OLS-CUSUM test does not reject stability (p = 0.36), but I acknoeldge now that it is quite a weak test here because it has little power to detect the kind of gradual, partly offsetting shifts the rolling plot reveals. The Chow tests, which look directly for a change at a chosen date, are more informative and reject stability at both end-2010 (p = 0.016) and end-2015 (p = 0.006). Taken together, the weight of the evidence is that the betas are not stationary.

This has a direct bearing on risk because, if the exposures drift, any risk figure calculated from the fixed full-sample betas could be misleading at any given moment. A fund that looks roughly value-neutral on average is in fact heavily short value/growth in its later years, so a hedge or a value-at-risk estimate built on the static loadings would understate the true exposure in those periods.