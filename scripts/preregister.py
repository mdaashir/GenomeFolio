"""Write a locked pre-registration spec (M10).

Generalises the two ad-hoc pre-registrations: it freezes the universe, the arms, the
primary comparison and the decision rule (plus a hash of params.yaml and the git commit)
into ``reports/preregistration_<name>.json`` *before* the run. ``run_preregistered.py``
then evaluates it once.

Usage::

    uv run python scripts/preregister.py smoke --universe alt --primary pooled_mnl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

from genomefolio.config import REPO_ROOT

ORIGINAL = {
    "tickers": ["AAPL", "MSFT", "INTC", "CSCO", "JPM", "VZ", "T", "XOM", "CVX", "CAT",
                "JNJ", "PFE", "MRK", "PG", "KO", "WMT", "DIS", "HD", "MCD", "NKE"],
    "start": "2013-01-01", "end": "2025-01-01", "cache_dir": "data/cache", "fixture": None,
}
HOLDOUT = {
    "tickers": ["NVDA", "AMD", "QCOM", "TXN", "ORCL", "ADBE", "CRM", "IBM", "ABT", "BMY",
                "LLY", "AMGN", "GILD", "TGT", "LOW", "SBUX", "BA", "GE", "MMM", "HON"],
    "start": "2013-01-01", "end": "2025-01-01", "cache_dir": "data/cache_holdout", "fixture": None,
}
ALT = {
    "tickers": ["ORCL", "ADBE", "CRM", "QCOM", "TXN", "IBM", "ABT", "BMY", "LLY", "TGT", "LOW", "SBUX"],
    "start": "2019-01-01", "end": "2024-01-01", "cache_dir": "data/cache_alt",
    "fixture": "data/sample/prices_alt.csv",
}
UNIVERSES = {"original": ORIGINAL, "holdout": HOLDOUT, "alt": ALT}


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--universe", choices=list(UNIVERSES), default="alt")
    ap.add_argument("--arms", default="pooled_mnl,per_stock")
    ap.add_argument("--reference", default="equal_weight")
    ap.add_argument("--primary", default="pooled_mnl")
    ap.add_argument("--comparison", default="higher_sharpe")
    ap.add_argument("--n-trials", type=int, default=2)
    args = ap.parse_args()

    params = (REPO_ROOT / "config" / "params.yaml").read_text(encoding="utf-8")
    spec = {
        "name": args.name,
        "universe": args.universe,
        "universe_spec": UNIVERSES[args.universe],
        "arms": args.arms.split(","),
        "reference": args.reference,
        "primary": args.primary,
        "comparison": args.comparison,
        "n_trials": args.n_trials,
        "params_sha256_8": hashlib.sha256(params.encode("utf-8")).hexdigest()[:8],
        "git_commit": _git_commit(),
    }
    path = REPO_ROOT / "reports" / f"preregistration_{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
