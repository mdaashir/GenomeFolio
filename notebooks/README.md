# Notebooks (exploration only)

Throwaway exploration lives here — dendrogram inspection, band-sweep stability, the
null-silhouette distribution, and quick sanity checks. Nothing in `notebooks/` is
imported by `src/` or covered by tests; promote anything durable into `src/`.

Suggested starting points:

- cluster dendrogram + silhouette-vs-k curve
- null vs real silhouette distribution (how far above the 95th percentile)
- turnover by rebalance, and where the gate failed
