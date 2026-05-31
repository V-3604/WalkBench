"""
Walk Score construct-validity check for WalkBench.

Question this answers: does WalkBench's composite walkability index (PC1 of five
visually grounded components) agree with the commercial Walk Score? This is a
one-shot sanity check, NOT a prediction target. We never train against Walk
Score -- doing so would re-open an apples-to-oranges comparison against prior
single-city pooled work. Here we only ask whether the two measures rank places
the same way.

Protocol:
- Draw a reproducible random sample of locked points, stratified across the four
  cities (default 750 total).
- Query the official Walk Score API for each sampled point's lat/lon.
- Compute Spearman rho between `walkability_index_v2` and Walk Score, overall and
  per city, with bootstrap 95% CIs.
- Apply the PRE-REGISTERED decision rule (see SRC_PLAN.md "Locked decisions"):
      overall rho >= --feature-threshold (default 0.50) AND p < 0.05
          -> verdict FEATURE        (report the correlation in the abstract)
      otherwise
          -> verdict DO_NOT_FEATURE (purge from paper + shared docs; defend the
             index on its components instead; mention divergence only in
             limitations)

Cost / time:
- The Walk Score API is free within a registered key's quota (default tier is
  5,000 calls/day). 750 points => $0.00 and ~4 min at the default 0.3 s pacing.
- Requires WALKSCORE_API_KEY in .env (register at walkscore.com/professional).
  The API lists `address` as required but scores from lat/lon; we pass a
  "lat,lon" string for the address field.

Run from repo root (Windows venv):

  Dry run (no API calls; prints sampling plan + cost/time estimate):
    & ".venv\\Scripts\\python.exe" project\\scripts\\correlate_walkscore.py --dry-run

  Smoke test (10 points, real API):
    & ".venv\\Scripts\\python.exe" project\\scripts\\correlate_walkscore.py --limit 10

  Full check (750 points, default):
    & ".venv\\Scripts\\python.exe" project\\scripts\\correlate_walkscore.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from tqdm import tqdm

WALKSCORE_ENDPOINT = "https://api.walkscore.com/score"
CITIES = ("msp", "seattle", "dc", "pittsburgh")
INDEX_COLUMN = "walkability_index_v2"
SEED = 42


# ---------------------------------------------------------------------------
# Repo / env helpers (mirrors generate_openai_structured_captions.py)
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "project").exists():
            return p
    raise RuntimeError("Could not locate repo root (expected a 'project/' folder).")


def load_dotenv_if_present(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in __import__("os").environ:
            __import__("os").environ[key] = value


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SamplePoint:
    point_id: int
    city: str
    lat: float
    lon: float
    index_value: float


@dataclass
class FetchedScore:
    point_id: int
    city: str
    index_value: float
    walkscore: float
    status: int


@dataclass
class Correlation:
    label: str
    n: int
    rho: float
    p_value: float
    ci_low: float
    ci_high: float


@dataclass
class ValidationReport:
    written_at: str
    seed: int
    requested: int
    n_success: int
    n_failed: int
    feature_threshold: float
    overall: Correlation
    per_city: list[Correlation]
    verdict: str
    failure_reasons: dict[str, int]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def load_index_frame(labels_csv: Path, index_csv: Path) -> pd.DataFrame:
    """Return a frame with point_id, city, lat, lon, INDEX_COLUMN.

    The index column lives in the master training CSV in the current layout; if
    a future layout moves it out, fall back to merging walkability_index.csv.
    """
    df = pd.read_csv(labels_csv, low_memory=False)
    needed = {"point_id", "city", "lat", "lon"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"{labels_csv.name} is missing required columns {sorted(missing)}. "
            "Expected the master features_labels_agreement.csv schema."
        )
    if INDEX_COLUMN not in df.columns:
        if not index_csv.exists():
            raise FileNotFoundError(
                f"{INDEX_COLUMN} not in {labels_csv.name} and {index_csv.name} not "
                f"found. Run build_walkability_index_src.py first."
            )
        idx = pd.read_csv(index_csv)[["point_id", "city", INDEX_COLUMN]]
        df = df.merge(idx, on=["point_id", "city"], how="left")
    out = df[["point_id", "city", "lat", "lon", INDEX_COLUMN]].copy()
    out = out.dropna(subset=["lat", "lon", INDEX_COLUMN])
    return out


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> list[SamplePoint]:
    rng = random.Random(seed)
    cities = [c for c in CITIES if c in set(df["city"])]
    per_city = max(1, sample_size // len(cities))
    points: list[SamplePoint] = []
    for city in cities:
        rows = df[df["city"] == city]
        take = min(per_city, len(rows))
        chosen = rng.sample(range(len(rows)), take)
        sub = rows.iloc[chosen]
        for r in sub.itertuples(index=False):
            points.append(
                SamplePoint(
                    point_id=int(r.point_id),
                    city=str(r.city),
                    lat=float(r.lat),
                    lon=float(r.lon),
                    index_value=float(getattr(r, INDEX_COLUMN)),
                )
            )
    rng.shuffle(points)
    return points


# ---------------------------------------------------------------------------
# Walk Score API
# ---------------------------------------------------------------------------

def fetch_walkscore(
    session: requests.Session, api_key: str, point: SamplePoint, timeout: float
) -> tuple[float | None, int, str | None]:
    """Return (walkscore, status, error). walkscore is None on any failure."""
    params = {
        "format": "json",
        "lat": point.lat,
        "lon": point.lon,
        "address": f"{point.lat},{point.lon}",
        "wsapikey": api_key,
    }
    resp = session.get(WALKSCORE_ENDPOINT, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    status = int(data.get("status", -1))
    if status == 1 and data.get("walkscore") is not None:
        return float(data["walkscore"]), status, None
    reason = {
        2: "score_pending",
        30: "invalid_latlon",
        31: "invalid_key",
        40: "quota_exceeded",
        41: "daily_quota_exceeded",
        42: "ip_blocked",
    }.get(status, f"status_{status}_or_no_score")
    return None, status, reason


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def spearman_with_ci(
    x: np.ndarray, y: np.ndarray, label: str, n_boot: int, seed: int
) -> Correlation:
    res = stats.spearmanr(x, y)
    rho = float(res.statistic)
    p_value = float(res.pvalue)
    rng = np.random.default_rng(seed)
    n = len(x)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = stats.spearmanr(x[idx], y[idx]).statistic
    ci_low, ci_high = np.percentile(boot[~np.isnan(boot)], [2.5, 97.5])
    return Correlation(label, n, rho, p_value, float(ci_low), float(ci_high))


def decide(overall: Correlation, threshold: float) -> str:
    if overall.rho >= threshold and overall.p_value < 0.05:
        return "FEATURE"
    return "DO_NOT_FEATURE"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-size", type=int, default=750, help="Total points across all cities (default 750).")
    ap.add_argument("--limit", type=int, default=None, help="Cap total API calls (smoke test); overrides --sample-size.")
    ap.add_argument("--dry-run", action="store_true", help="Print sampling plan + cost/time estimate; make no API calls.")
    ap.add_argument("--feature-threshold", type=float, default=0.50, help="Min overall Spearman rho to FEATURE in the paper (default 0.50).")
    ap.add_argument("--min-success", type=float, default=0.80, help="Min fraction of successful API calls before emitting a verdict (default 0.80).")
    ap.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for CIs (default 1000).")
    ap.add_argument("--sleep", type=float, default=0.3, help="Seconds between API calls for rate-limit politeness (default 0.3).")
    ap.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout in seconds (default 20).")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info(f"Seed = {args.seed}")

    root = repo_root()
    load_dotenv_if_present(root / ".env")
    labels_csv = root / "project" / "data" / "processed" / "labels" / "features_labels_agreement.csv"
    index_csv = root / "project" / "data" / "processed" / "labels" / "walkability_index.csv"
    out_path = root / "project" / "artifacts" / "reports" / "walkscore_validation.json"

    df = load_index_frame(labels_csv, index_csv)
    sample_size = args.limit if args.limit is not None else args.sample_size
    points = stratified_sample(df, sample_size, args.seed)

    counts: dict[str, int] = {}
    for pt in points:
        counts[pt.city] = counts.get(pt.city, 0) + 1
    est_seconds = len(points) * (args.sleep + 0.15)
    logging.info(f"Sampled {len(points)} points: " + ", ".join(f"{c}={n}" for c, n in sorted(counts.items())))
    logging.info(f"Estimated wall time: ~{est_seconds / 60:.1f} min  |  Estimated cost: $0.00 (free Walk Score quota)")

    if args.dry_run:
        logging.info("Dry run: no API calls made. Decision rule: FEATURE if overall rho >= "
                     f"{args.feature_threshold} and p < 0.05, else DO_NOT_FEATURE.")
        return 0

    import os
    api_key = os.environ.get("WALKSCORE_API_KEY")
    if not api_key:
        raise SystemExit(
            "WALKSCORE_API_KEY not set. Add it to .env (register at "
            "walkscore.com/professional) and re-run."
        )

    session = requests.Session()
    fetched: list[FetchedScore] = []
    failure_reasons: dict[str, int] = {}
    for pt in tqdm(points, desc="Walk Score"):
        score, status, error = fetch_walkscore(session, api_key, pt, args.timeout)
        if error is None:
            fetched.append(FetchedScore(pt.point_id, pt.city, pt.index_value, score, status))
        else:
            failure_reasons[error] = failure_reasons.get(error, 0) + 1
        time.sleep(args.sleep)

    n_success = len(fetched)
    n_failed = len(points) - n_success
    success_frac = n_success / len(points) if points else 0.0
    logging.info(f"Fetched {n_success}/{len(points)} ({success_frac:.1%}); failures: {failure_reasons or 'none'}")
    if success_frac < args.min_success:
        raise SystemExit(
            f"Only {success_frac:.1%} of calls succeeded (< --min-success {args.min_success}). "
            f"Refusing to emit a verdict from sparse data. Reasons: {failure_reasons}"
        )

    fdf = pd.DataFrame([asdict(f) for f in fetched])
    overall = spearman_with_ci(
        fdf["index_value"].to_numpy(),
        fdf["walkscore"].to_numpy(),
        "overall", args.n_boot, args.seed,
    )
    per_city: list[Correlation] = []
    for city in sorted(fdf["city"].unique()):
        sub = fdf[fdf["city"] == city]
        if len(sub) >= 10:
            per_city.append(
                spearman_with_ci(sub["index_value"].to_numpy(), sub["walkscore"].to_numpy(),
                                 city, args.n_boot, args.seed)
            )

    verdict = decide(overall, args.feature_threshold)
    report = ValidationReport(
        written_at=datetime.now(timezone.utc).isoformat(),
        seed=args.seed,
        requested=len(points),
        n_success=n_success,
        n_failed=n_failed,
        feature_threshold=args.feature_threshold,
        overall=overall,
        per_city=per_city,
        verdict=verdict,
        failure_reasons=failure_reasons,
    )
    out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    logging.info("")
    logging.info("=" * 64)
    logging.info(f"  overall Spearman rho = {overall.rho:+.3f}  "
                 f"(95% CI {overall.ci_low:+.3f}..{overall.ci_high:+.3f}, p={overall.p_value:.2e}, n={overall.n})")
    for c in per_city:
        logging.info(f"    {c.label:<12} rho = {c.rho:+.3f}  (n={c.n})")
    logging.info("-" * 64)
    if verdict == "FEATURE":
        logging.info(f"  VERDICT: FEATURE  (rho >= {args.feature_threshold} and significant)")
        logging.info("  -> Report one line in the abstract: the transparent component-based")
        logging.info(f"     index correlates rho={overall.rho:.2f} with the commercial Walk Score.")
    else:
        logging.info(f"  VERDICT: DO_NOT_FEATURE  (rho {overall.rho:.2f} < {args.feature_threshold} or not significant)")
        logging.info("  -> PURGE Walk Score from the paper and shared docs. Defend the index on")
        logging.info("     its five components; note the divergence only in limitations.")
    logging.info("=" * 64)
    logging.info(f"Report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
