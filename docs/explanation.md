# GenomeFolio — Project Explanation

## 1. The problem

We want to group 15–25 stocks by the **similarity of their return patterns**, not by
correlation, and then forecast how each group behaves. Correlation answers "do these two
series move together over the whole window?" It does not answer "do these two series show
the same shape, even if one is a few days ahead of the other?"

GenomeFolio borrows the idea of sequence alignment from bioinformatics. Each stock's daily
returns become a string of symbols; two strings are compared by how well they align, gaps
included. Stocks whose strings align well are clustered. Each cluster's composite return is
then forecast over the next month.

## 2. Why sequence alignment, not correlation

- Correlation is a single number over a whole window and is dominated by magnitude and
  co-movement. It cannot reward a pattern that repeats with a small time shift.
- Global alignment can match symbols that are a few positions apart by inserting gaps, so a
  pattern that appears two days later in one stock still aligns.
- The alignment uses a **graded substitution matrix**: a near-miss (big-up versus small-up)
  costs less than an opposite (big-up versus big-down). Correlation has no notion of
  near-miss.
- A **band** limits how far a gap may shift the alignment, so the method cannot pair
  unrelated days just to raise the score.

## 3. Stage 1 — Data

Daily adjusted close prices are downloaded once and cached under `data/` (gitignored). Prices
are converted to simple daily returns:

```
r_t = (P_t / P_(t-1)) - 1
```

Only dates on which every stock trades are kept, so all symbol strings share one index.
A committed 10-stock, five-year sample under `data/sample/` lets the tests and the offline
demo run without network access.

## 4. Stage 2 — Symbolization

For each stock, four **training-window quantiles** (default 20/40/60/80 percentiles) cut its
returns into five symbols:

- `0` big-down, `1` small-down, `2` flat, `3` small-up, `4` big-up.

Because the thresholds are the stock's own quantiles, every symbol occurs about 20% of the
time. This deliberately discards magnitude: the strings encode *when* moves happen, not how
big they are. Two stocks with very different volatility become comparable.

Edge cases:

- A constant series has no ordering and is emitted as all `flat`.
- Missing values stay missing (the pipeline removes them upstream).
- `quantile_scope: pooled` computes one set of thresholds from all stocks pooled, which
  preserves relative volatility instead (an optional ablation).
- `market_neutral: true` symbolizes the stock return minus the equal-weight universe return.

## 5. Stage 3 — Alignment

Two symbol strings are aligned with **banded global alignment** (Needleman-Wunsch) and
**affine gaps**. The substitution matrix is graded and symmetric:

```
S(s, t) = match_score        if s == t
        = -|s - t|           otherwise
```

The dynamic program keeps three layers — `M` (aligned pair), `X` (gap in the second string)
and `Y` (gap in the first):

```
M[i,j] = max(M[i-1,j-1], X[i-1,j-1], Y[i-1,j-1]) + S[a_i, b_j]
X[i,j] = max(M[i-1,j] + open + extend, X[i-1,j] + extend)
Y[i,j] = max(M[i,j-1] + open + extend, Y[i,j-1] + extend)
```

Only cells with `|i - j| <= band` are computed. A gap costs `open` when it starts and
`extend` per additional position, so one clean shift is cheaper than many scattered gaps.
The band must be at least the difference in string lengths, or the end cell is unreachable
and the function raises.

The raw score is normalised by the two self-scores:

```
similarity(a, b) = score(a, b) / sqrt(self(a) * self(b))
distance(a, b)   = 1 - similarity(a, b)
```

This distance is **not claimed to be a metric**: it can violate the triangle inequality. The
code asserts only that the matrix is symmetric and non-negative, which is all average
linkage requires. Smith-Waterman local alignment is available as a cross-check only.

## 6. Stage 4 — Clustering

The distance matrix is clustered with **hierarchical average linkage**. The number of
clusters is not fixed; it is chosen by silhouette over a constrained range:

- `k` must lie in `[k_min, min(k_max, floor(N / k_cap_divisor))]`.
- Every cluster must contain at least `min_cluster_size` stocks.
- The cut with the best silhouette among valid candidates wins.
- If no cut satisfies the size rule, the best-silhouette cut is returned with a flag
  (`min_size_ok = False`), and the CLI prints a note.

A **band sweep** re-runs the clustering at several band widths (`2, 5, 10`) and reports
Adjusted Rand Index stability, so the chosen solution is not an artefact of one band.

## 7. Stage 5 — Forecasting

For each cluster, the equal-weight composite return series is built from its members. The
composite is forecast over the next 21 trading days (one month):

- `d` (differencing order) is fixed by an Augmented Dickey-Fuller test on the training
  window: 0 if the series is already stationary, otherwise 1.
- `p` and `q` are chosen by AIC over a small grid with `p, q <= max_p / max_q`.
- The forecast steps are summed to give the **cumulative expected return** over the horizon.
- If no model converges, or the series is degenerate, the historical mean is used and the
  result is flagged as a fallback.
- A smoothing-spline extrapolation of the cumulative return is available as an alternative
  (`forecast.model: spline`).

Daily returns are only weakly predictable, so the forecast is a modest signal and is always
reported with its horizon.

## 8. Design decisions

- **Per-stock quantiles** rather than fixed percentage bands: makes strings comparable across
  stocks of different volatility.
- **Graded matrix** rather than identity-only matching: near-misses should cost less than
  opposites.
- **Affine gaps plus a band** rather than linear gaps: encourages one meaningful shift and
  prevents pairing unrelated days.
- **Constrained k with a minimum cluster size**: raw silhouette tends to pick k = 2 for
  around 20 stocks, which is not informative.
- **Stop at the forecast**: the horizon is explicit and testable. Portfolio construction and
  backtesting are deliberately out of scope in this version.

## 9. No look-ahead

Every stage at time *t* uses only data up to *t*:

- symbol thresholds are quantiles of the trailing training window;
- alignment and clustering use only that window;
- the forecast is fitted on the window and projects forward.

The CLI always takes the most recent `train_days` window and never uses future returns.

## 10. Limitations

- **Weak predictability**: daily returns are close to noise; ARIMA is close to the mean.
- **Window sensitivity**: a different training window produces different thresholds and
  clusters; check the band sweep before trusting one configuration.
- **Survivorship bias**: the fixed universe contains names that survived to the present.
- **Small samples**: 15–25 stocks give a coarse clustering; the constrained search is a
  guard, not a proof.
- **No significance gate**: the pipeline trusts the clustering as given. Any claim that the
  clusters are meaningful should be checked separately.
