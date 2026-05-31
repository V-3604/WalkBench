"""
Generate structured walkability captions using OpenAI vision models.

For each intersection point this script:
- Assembles a 2×2 composite from the 4 heading images (0/90/180/270)
- Sends the composite to an OpenAI vision model
- Gets back a strict JSON object with 10 walkability labels
- Writes results to JSONL + CSV (same schema as the Gemini script)

Supports two processing modes:
  realtime  - standard Chat Completions API (immediate, higher RPM limit)
  batch     - OpenAI Batch API (50% cheaper, async, up to 24 h turnaround)

Supported cities:
  msp       - project/data/raw/imagery/street_view_mapillary/Minneapolis/
  seattle   - project/data/raw/imagery/street_view_mapillary/Seattle/
  both      - run MSP then Seattle

---
Model recommendation
---------------------
gpt-5.4-nano  ← DEFAULT — newer architecture (Aug 2025 cutoff), purpose-built for
               classification and data extraction, same price tier as gpt-4o-mini.
               Model string: "gpt-5.4-nano"

gpt-4o-mini   — fallback if gpt-5.4-nano unavailable on your tier.
               Model string: "gpt-4o-mini"

---
Image counts and cost estimates (gpt-5.4-nano, ~$0.20/$1.25 per 1M in/out)
----------------------------------------------------------------------------
Points per city:           ~4,874
Input tokens per call:     ~2,200  (image high-detail + enriched prompt)
Output tokens per call:    ~150    (10-field JSON)

                Standard cost    Batch cost (-50%)
gpt-5.4-nano    ~$2.60           ~$1.30    ← recommended
gpt-4o-mini     ~$3.50           ~$1.75    (older model, larger prompt tokens)

---
Run from repo root:

  Smoke test (5 Seattle points, realtime):
    & "C:\\Users\\kvars\\Desktop\\WalkCLIP\\.venv\\Scripts\\python.exe" project\\scripts\\generate_openai_structured_captions.py --city seattle --limit 5

  Full Seattle, realtime:
    & "C:\\Users\\kvars\\Desktop\\WalkCLIP\\.venv\\Scripts\\python.exe" project\\scripts\\generate_openai_structured_captions.py --city seattle

  Both cities, Batch API (cheapest):
    & "C:\\Users\\kvars\\Desktop\\WalkCLIP\\.venv\\Scripts\\python.exe" project\\scripts\\generate_openai_structured_captions.py --city both --mode batch

  Print cost estimate from existing JSONL:
    & "C:\\Users\\kvars\\Desktop\\WalkCLIP\\.venv\\Scripts\\python.exe" project\\scripts\\generate_openai_structured_captions.py --city seattle --print-cost
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Repo / env helpers
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
    try:
        for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        return


def get_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if not key:
        raise SystemExit(
            "Missing OpenAI API key.\n"
            "Set OPENAI_API_KEY=sk-... in your .env or environment."
        )
    return key


# ---------------------------------------------------------------------------
# City config
# ---------------------------------------------------------------------------

@dataclass
class CityConfig:
    name: str
    street_view_dir: Path
    out_jsonl: Path
    out_csv: Path


def build_city_configs(root: Path) -> dict[str, CityConfig]:
    text_dir = root / "project" / "data" / "processed" / "text"
    return {
        "msp": CityConfig(
            name="msp",
            street_view_dir=(
                root / "project" / "data" / "raw" / "imagery"
                / "street_view_mapillary" / "Minneapolis"
            ),
            out_jsonl=text_dir / "streetview_openai_msp.jsonl",
            out_csv=text_dir / "streetview_openai_msp.csv",
        ),
        "seattle": CityConfig(
            name="seattle",
            street_view_dir=(
                root / "project" / "data" / "raw" / "imagery"
                / "street_view_mapillary" / "Seattle"
            ),
            out_jsonl=text_dir / "streetview_openai_seattle.jsonl",
            out_csv=text_dir / "streetview_openai_seattle.csv",
        ),
        "dc": CityConfig(
            name="dc",
            street_view_dir=(
                root / "project" / "data" / "raw" / "imagery"
                / "street_view_mapillary" / "DC"
            ),
            out_jsonl=text_dir / "streetview_openai_dc.jsonl",
            out_csv=text_dir / "streetview_openai_dc.csv",
        ),
    }


# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------

ALLOWED: dict[str, list[str]] = {
    "sidewalk":       ["none", "one_side", "both_sides", "unknown"],
    "crosswalk":      ["present", "absent", "unknown"],
    "bike_lane":      ["present", "absent", "unknown"],
    "traffic_volume": ["low", "med", "high", "unknown"],
    "tree_canopy":    ["none", "sparse", "moderate", "dense", "unknown"],
    "transit_stop":   ["present", "absent", "unknown"],
    "lighting":       ["good", "poor", "unknown"],
    "maintenance":    ["good", "fair", "poor", "unknown"],
}

ALLOWED_MULTI: dict[str, list[str]] = {
    "land_use": ["residential", "commercial", "industrial", "mixed", "unknown"],
    "barriers": ["highway", "rail", "river", "fence", "none", "unknown"],
}

CANONICAL_ORDER = [
    "sidewalk", "crosswalk", "bike_lane", "traffic",
    "tree_canopy", "land_use", "transit_stop", "lighting", "maintenance", "barriers",
]

# ---------------------------------------------------------------------------
# Prompt  (enriched for quality — adds domain context, decision rules,
#          and 3 few-shot text examples covering common ambiguous cases)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert urban-planning analyst labeling street-level imagery "
    "for a walkability research dataset. Your labels will be used directly in "
    "a machine-learning model, so precision and consistency matter more than speed. "
    "Apply the decision rules below strictly and uniformly."
)

def build_prompt() -> str:
    return """\
You are labeling a 2×2 COMPOSITE of four street-view photos taken at the SAME \
intersection, facing North (top-left), East (top-right), South (bottom-left), \
and West (bottom-right).

DECISION RULES — read carefully before labeling:

sidewalk:
  "none"       — no paved pedestrian path visible in ANY heading.
  "one_side"   — sidewalk present on only one side of the street in the majority of headings.
  "both_sides" — sidewalk on BOTH sides visible in the majority of headings.
  Use the dominant pattern across all four views; ignore one anomalous heading.

crosswalk:
  "present" — painted stripes OR raised crossing OR tactile paving visible at or near the intersection.
  "absent"  — no crossing markings visible despite clear view of the pavement.
  "unknown" — pavement obscured, angle unclear, or intersection not visible.

bike_lane:
  "present" — dedicated painted lane, shared-lane marking (sharrow), or separated cycle track visible.
  "absent"  — no bike infrastructure visible on any heading.

traffic_volume (judge by lane count, parked cars, and visible vehicle density):
  "low"  — ≤2 travel lanes, quiet residential feel, few or no moving vehicles.
  "med"  — 2–4 lanes or moderate vehicle presence.
  "high" — ≥4 lanes, arterial road, heavy vehicle presence or bus/truck activity.

tree_canopy (judge the overhead canopy over the street and sidewalk):
  "none"     — no trees or only bare stumps.
  "sparse"   — occasional trees, canopy covers <25% of the street corridor.
  "moderate" — trees present, 25–60% coverage.
  "dense"    — continuous canopy, >60% coverage.

land_use (may be a list; pick all that clearly apply):
  "residential"  — houses, apartments, condos visible as the primary use.
  "commercial"   — storefronts, offices, restaurants, retail signage.
  "industrial"   — warehouses, factories, loading docks, industrial signage.
  "mixed"        — clear combination of two or more uses in the SAME building or block.
  "unknown"      — insufficient visual information.

transit_stop:
  "present" — bus stop sign, shelter, bench with transit markings, or rail station entrance visible.
  "absent"  — no transit infrastructure visible in any heading.

lighting:
  "good" — street lights visible on both sides or high-mast lights; even coverage implied.
  "poor" — no street lights visible, or lights obviously damaged/missing on most headings.
  "unknown" — daytime image with no lights on; cannot determine.

maintenance:
  "good" — pavement intact, sidewalks uncracked, no visible graffiti or debris.
  "fair" — minor cracks, patching, or surface wear; functional but imperfect.
  "poor" — large potholes, broken sidewalks, overgrown vegetation, visible neglect.

barriers (may be a list; list ALL that clearly apply):
  "highway" — elevated or grade-separated highway structure adjacent.
  "rail"    — active or inactive rail tracks crossing or adjacent.
  "river"   — visible water body forming a physical boundary.
  "fence"   — tall fence, wall, or sound barrier cutting off pedestrian access.
  "none"    — no physical barriers to pedestrian movement; output ["none"].

---
FEW-SHOT EXAMPLES (text descriptions of what the model saw, not images):

Example 1 — Suburban residential street, wide, mostly car-oriented:
  Observation: All four headings show a wide 4-lane arterial with a grass strip \
but no sidewalk on the south side and a cracked sidewalk on the north side. \
No crosswalk markings. No bike infrastructure. Low tree coverage. \
Detached single-family houses set back from road. One bus stop sign visible \
facing east. Street lights on one side only. Surface has patches but no potholes.
  Output:
  {"sidewalk":"one_side","crosswalk":"absent","bike_lane":"absent",\
"traffic_volume":"high","tree_canopy":"sparse","land_use":["residential"],\
"transit_stop":"present","lighting":"poor","maintenance":"fair","barriers":["none"]}

Example 2 — Dense urban commercial intersection:
  Observation: All four headings show a tight urban grid. Sidewalks on both \
sides of all streets, painted crosswalks on all four arms of the intersection. \
Sharrow markings visible on two headings. 2-lane road, parked cars lining the \
curb. Storefront retail at ground level, apartments above — same buildings. \
No bus stop visible. Street lights present on all four corners. Pavement clean \
and intact, minor sidewalk crack on one heading only.
  Output:
  {"sidewalk":"both_sides","crosswalk":"present","bike_lane":"present",\
"traffic_volume":"low","tree_canopy":"sparse","land_use":["commercial","residential"],\
"transit_stop":"absent","lighting":"good","maintenance":"good","barriers":["none"]}

Example 3 — Industrial area near freight rail:
  Observation: Headings show a wide industrial road with no sidewalks on \
any heading. No crosswalk. No bike lane. Active rail tracks cross the \
intersection on the north-south axis. Large warehouse buildings. \
Heavy truck presence. No trees. One industrial streetlight visible but \
appears to be for a parking lot, not the street. Pavement severely potholed.
  Output:
  {"sidewalk":"none","crosswalk":"absent","bike_lane":"absent",\
"traffic_volume":"high","tree_canopy":"none","land_use":["industrial"],\
"transit_stop":"absent","lighting":"poor","maintenance":"poor","barriers":["rail"]}

---
Now label the composite image provided.

Return ONLY a strict JSON object — no markdown, no prose, no extra keys.
Every key is required. Use "unknown" if a label cannot be determined from the image.
For list fields (land_use, barriers), always output a JSON array.
"""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

HEADINGS = ["0", "90", "180", "270"]


def composite_bytes(street_view_dir: Path, pid: int) -> bytes:
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit(f"Pillow is required.  pip install pillow\nOriginal: {e}")

    paths = {h: street_view_dir / f"{pid}_{h}.jpg" for h in HEADINGS}
    missing = [h for h, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing headings for id={pid}: {missing}")

    imgs = {h: Image.open(paths[h]).convert("RGB") for h in HEADINGS}
    w, h_ = imgs["0"].size
    canvas = Image.new("RGB", (w * 2, h_ * 2))
    canvas.paste(imgs["0"],   (0,  0))
    canvas.paste(imgs["90"],  (w,  0))
    canvas.paste(imgs["180"], (0,  h_))
    canvas.paste(imgs["270"], (w,  h_))

    buf = BytesIO()
    canvas.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()


def discover_ids(street_view_dir: Path) -> list[int]:
    rx = re.compile(r"^(\d+)_(0|90|180|270)\.jpg$", re.IGNORECASE)
    by_id: dict[int, set[str]] = {}
    for p in street_view_dir.glob("*.jpg"):
        m = rx.match(p.name)
        if not m:
            continue
        pid = int(m.group(1))
        by_id.setdefault(pid, set()).add(m.group(2))
    needed = {"0", "90", "180", "270"}
    return sorted(pid for pid, have in by_id.items() if needed.issubset(have))


def read_requested_ids(ids_path: Path | None) -> set[int] | None:
    if ids_path is None:
        return None
    requested: set[int] = set()
    with ids_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not line.isdigit():
                raise ValueError(f"Non-numeric point_id in {ids_path}: {line!r}")
            requested.add(int(line))
    return requested


def restrict_ids(
    ids: list[int],
    requested_ids: set[int] | None,
    *,
    city_name: str,
    source_label: str,
) -> list[int]:
    if requested_ids is None:
        return ids
    available = set(ids)
    filtered = [pid for pid in ids if pid in requested_ids]
    missing = sorted(requested_ids - available)
    print(
        f"[{city_name}] {len(filtered)} IDs selected from {source_label}; "
        f"{len(missing)} requested IDs are not complete on disk."
    )
    if missing:
        preview = ", ".join(str(pid) for pid in missing[:15])
        suffix = " ..." if len(missing) > 15 else ""
        print(f"[{city_name}] Missing from current imagery: {preview}{suffix}")
    return filtered


def with_output_suffix(city_cfg: CityConfig, suffix: str) -> CityConfig:
    if not suffix:
        return city_cfg
    return replace(
        city_cfg,
        out_jsonl=city_cfg.out_jsonl.with_name(f"{city_cfg.out_jsonl.stem}{suffix}{city_cfg.out_jsonl.suffix}"),
        out_csv=city_cfg.out_csv.with_name(f"{city_cfg.out_csv.stem}{suffix}{city_cfg.out_csv.suffix}"),
    )


def read_done_ids(jsonl_path: Path) -> set[int]:
    done: set[int] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            pid = rec.get("id")
            if not isinstance(pid, int):
                continue
            if "error" in rec:
                continue
            if isinstance(rec.get("answer"), str) and isinstance(rec.get("json"), dict):
                done.add(pid)
    return done


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------

def _pick_single(name: str, v: Any) -> str:
    allowed = ALLOWED[name]
    return v if isinstance(v, str) and v in allowed else "unknown"


def _pick_multi(name: str, v: Any) -> list[str]:
    allowed_set = set(ALLOWED_MULTI[name])
    items = [v] if isinstance(v, str) else (v if isinstance(v, list) else ["unknown"])
    cleaned = [x for x in items if isinstance(x, str) and x in allowed_set]
    if not cleaned:
        cleaned = ["unknown"]
    if name == "barriers" and "none" in cleaned:
        return ["none"]
    order = ALLOWED_MULTI[name]
    return sorted(set(cleaned), key=lambda x: order.index(x))


def validate_and_canonicalize(js: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out: dict[str, Any] = {
        "sidewalk":       _pick_single("sidewalk",       js.get("sidewalk")),
        "crosswalk":      _pick_single("crosswalk",      js.get("crosswalk")),
        "bike_lane":      _pick_single("bike_lane",      js.get("bike_lane")),
        "traffic_volume": _pick_single("traffic_volume", js.get("traffic_volume")),
        "tree_canopy":    _pick_single("tree_canopy",    js.get("tree_canopy")),
        "land_use":       _pick_multi("land_use",        js.get("land_use")),
        "transit_stop":   _pick_single("transit_stop",   js.get("transit_stop")),
        "lighting":       _pick_single("lighting",       js.get("lighting")),
        "maintenance":    _pick_single("maintenance",    js.get("maintenance")),
        "barriers":       _pick_multi("barriers",        js.get("barriers")),
    }

    def fmt(v: Any) -> str:
        if isinstance(v, list):
            return v[0] if len(v) == 1 else "|".join(v)
        return v if isinstance(v, str) else "unknown"

    tags = {
        "sidewalk":     out["sidewalk"],
        "crosswalk":    out["crosswalk"],
        "bike_lane":    out["bike_lane"],
        "traffic":      out["traffic_volume"],
        "tree_canopy":  out["tree_canopy"],
        "land_use":     fmt(out["land_use"]),
        "transit_stop": out["transit_stop"],
        "lighting":     out["lighting"],
        "maintenance":  out["maintenance"],
        "barriers":     fmt(out["barriers"]),
    }
    canonical = ", ".join(f"{k}={tags[k]}" for k in CANONICAL_ORDER)
    return out, canonical


def _safe_load_json_object(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip().lstrip("`").rstrip("`").strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        obj = json.loads(cleaned[start: end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Model output is not a JSON object.")
    return obj


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, units_per_minute: float, units_per_request: float = 1.0) -> None:
        self._lock = threading.Lock()
        self._next_time = 0.0
        if units_per_minute <= 0 or units_per_request <= 0:
            self._interval = 0.0
        else:
            self._interval = 60.0 * units_per_request / units_per_minute

    def wait_turn(self) -> None:
        if self._interval <= 0:
            return
        sleep_s = 0.0
        with self._lock:
            now = time.time()
            if self._next_time <= now:
                self._next_time = now + self._interval
                return
            sleep_s = self._next_time - now
            self._next_time += self._interval
        if sleep_s > 0:
            time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# Realtime API
# ---------------------------------------------------------------------------

def _exp_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    return min(cap, base * (2 ** max(0, attempt - 1)))


@dataclass
class OpenAIResult:
    parsed: dict[str, Any]
    raw_text: str
    usage: dict[str, Any] | None = field(default=None)


def call_openai_realtime(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_bytes: bytes,
    timeout_s: float = 90.0,
    max_attempts: int = 10,
    detail: str = "high",
) -> OpenAIResult:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": detail,
                        },
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 512,
    }

    url = "https://api.openai.com/v1/chat/completions"
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout_s)
            if r.status_code == 200:
                resp = r.json()
                text = resp["choices"][0]["message"]["content"]
                usage = resp.get("usage")
                parsed = _safe_load_json_object(text)
                return OpenAIResult(parsed=parsed, raw_text=text, usage=usage)
            if r.status_code in (429, 500, 502, 503, 504):
                try:
                    retry_after = float(r.headers.get("Retry-After", _exp_backoff(attempt)))
                except Exception:
                    retry_after = _exp_backoff(attempt)
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                if attempt >= max_attempts:
                    raise last_err
                time.sleep(retry_after)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except (requests.RequestException, RuntimeError, json.JSONDecodeError, ValueError) as exc:
            last_err = exc
            if attempt >= max_attempts:
                raise
            time.sleep(_exp_backoff(attempt))

    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

BATCH_ENDPOINT = "/v1/chat/completions"


def _build_batch_request(
    pid: int, model: str, system_prompt: str, user_prompt: str,
    image_bytes: bytes, detail: str = "high"
) -> dict:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        "custom_id": str(pid),
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": detail,
                            },
                        },
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 512,
        },
    }


def _openai_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _upload_batch_file(api_key: str, jsonl_bytes: bytes, timeout_s: float = 120.0) -> str:
    r = requests.post(
        "https://api.openai.com/v1/files",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("batch_input.jsonl", jsonl_bytes, "application/json")},
        data={"purpose": "batch"},
        timeout=timeout_s,
    )
    if r.status_code != 200:
        raise RuntimeError(f"File upload failed {r.status_code}: {r.text[:400]}")
    return r.json()["id"]


def _create_batch(api_key: str, file_id: str, timeout_s: float = 60.0) -> str:
    r = requests.post(
        "https://api.openai.com/v1/batches",
        headers=_openai_headers(api_key),
        json={
            "input_file_id": file_id,
            "endpoint": BATCH_ENDPOINT,
            "completion_window": "24h",
        },
        timeout=timeout_s,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Batch create failed {r.status_code}: {r.text[:400]}")
    return r.json()["id"]


def _poll_batch(api_key: str, batch_id: str, poll_interval_s: int = 60) -> dict:
    url = f"https://api.openai.com/v1/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Batch poll failed {r.status_code}: {r.text[:300]}")
        obj = r.json()
        status = obj.get("status", "")
        done = obj.get("request_counts", {}).get("completed", 0)
        total = obj.get("request_counts", {}).get("total", "?")
        print(f"\r  batch {batch_id}: {status}  {done}/{total}", end="", flush=True)
        if status in terminal:
            print()
            return obj
        time.sleep(poll_interval_s)


def _download_batch_output(api_key: str, output_file_id: str, timeout_s: float = 300.0) -> list[dict]:
    r = requests.get(
        f"https://api.openai.com/v1/files/{output_file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout_s,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Output download failed {r.status_code}: {r.text[:300]}")
    return [json.loads(l) for l in r.text.splitlines() if l.strip()]


def _write_record(jf, cw, rec: dict, csv_exists_ref: list) -> None:
    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    jf.flush()
    if "answer" in rec and "error" not in rec:
        if not csv_exists_ref[0]:
            cw.writerow(["image", "answer"])
            csv_exists_ref[0] = True
        cw.writerow([rec["image"], rec["answer"]])


def run_batch_mode(
    *,
    api_key: str,
    city_cfg: CityConfig,
    model: str,
    system_prompt: str,
    user_prompt: str,
    ids: list[int],
    done: set[int],
    poll_interval_s: int = 60,
    detail: str = "high",
) -> None:
    todo = [pid for pid in ids if pid not in done]
    if not todo:
        print(f"[{city_cfg.name}] All {len(ids)} points already done. Nothing to batch.")
        return

    print(f"[{city_cfg.name}] Building batch JSONL for {len(todo)} points…")
    batch_lines: list[bytes] = []
    skipped = 0
    for pid in tqdm(todo, desc="Building batch", unit="pt"):
        try:
            img = composite_bytes(city_cfg.street_view_dir, pid)
        except FileNotFoundError:
            skipped += 1
            continue
        req = _build_batch_request(pid, model, system_prompt, user_prompt, img, detail=detail)
        batch_lines.append(json.dumps(req, ensure_ascii=False).encode("utf-8"))

    if not batch_lines:
        print(f"[{city_cfg.name}] No images found. Aborting batch.")
        return
    if skipped:
        print(f"[{city_cfg.name}] Skipped {skipped} points with incomplete headings.")

    jsonl_bytes = b"\n".join(batch_lines)
    print(f"[{city_cfg.name}] Uploading {len(batch_lines)} requests ({len(jsonl_bytes)//1024} KB)…")
    file_id = _upload_batch_file(api_key, jsonl_bytes)
    print(f"[{city_cfg.name}] File uploaded: {file_id}")
    batch_id = _create_batch(api_key, file_id)
    print(f"[{city_cfg.name}] Batch created: {batch_id}")
    print(f"[{city_cfg.name}] Polling every {poll_interval_s}s (up to 24 h)…")

    batch_obj = _poll_batch(api_key, batch_id, poll_interval_s=poll_interval_s)
    if batch_obj.get("status") != "completed":
        raise RuntimeError(
            f"Batch ended with status={batch_obj.get('status')}. "
            f"Error file: {batch_obj.get('error_file_id')}"
        )

    output_file_id = batch_obj.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("No output_file_id in completed batch object.")

    print(f"[{city_cfg.name}] Downloading results…")
    results = _download_batch_output(api_key, output_file_id)

    ensure_dirs(city_cfg.out_jsonl, city_cfg.out_csv)
    csv_exists_ref = [city_cfg.out_csv.exists()]
    written = 0
    with (
        city_cfg.out_jsonl.open("a", encoding="utf-8") as jf,
        city_cfg.out_csv.open("a", encoding="utf-8", newline="") as cf,
    ):
        cw = csv.writer(cf)
        if not csv_exists_ref[0]:
            cw.writerow(["image", "answer"])
            csv_exists_ref[0] = True

        for item in results:
            pid = int(item.get("custom_id", -1))
            resp_body = (item.get("response") or {}).get("body") or {}
            choices = resp_body.get("choices") or []
            usage = resp_body.get("usage")
            error = item.get("error")
            if error or not choices:
                rec = {
                    "id": pid, "image": f"{pid}_combined.jpg",
                    "error": str(error or "no choices"),
                    "model": model, "ts": datetime.now(timezone.utc).isoformat(),
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            text = choices[0]["message"]["content"]
            try:
                parsed = _safe_load_json_object(text)
                cleaned, canonical = validate_and_canonicalize(parsed)
                rec = {
                    "id": pid, "image": f"{pid}_combined.jpg",
                    "json": cleaned, "answer": canonical,
                    "model": model, "usage": usage,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cw.writerow([rec["image"], rec["answer"]])
                written += 1
            except Exception as exc:
                rec = {
                    "id": pid, "image": f"{pid}_combined.jpg",
                    "error": str(exc), "raw": text[:200],
                    "model": model, "ts": datetime.now(timezone.utc).isoformat(),
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[{city_cfg.name}] Wrote {written}/{len(results)} records.")


# ---------------------------------------------------------------------------
# Realtime mode runner
# ---------------------------------------------------------------------------

def run_realtime_mode(
    *,
    api_key: str,
    city_cfg: CityConfig,
    model: str,
    system_prompt: str,
    user_prompt: str,
    ids: list[int],
    done: set[int],
    workers: int = 8,
    max_rpm: float = 500.0,
    max_tpm: float = 0.0,
    estimated_tokens_per_request: int = 4250,
    timeout_s: float = 90.0,
    max_attempts: int = 10,
    detail: str = "high",
) -> None:
    todo = [pid for pid in ids if pid not in done]
    if not todo:
        print(f"[{city_cfg.name}] All {len(ids)} points already done.")
        return

    total = len(todo)
    effective_rpm = max_rpm
    if max_tpm > 0 and estimated_tokens_per_request > 0:
        tpm_bound_rpm = max_tpm / estimated_tokens_per_request
        effective_rpm = min(effective_rpm, tpm_bound_rpm) if effective_rpm > 0 else tpm_bound_rpm
    eta_msg = f" ETA~{(total / effective_rpm):.1f} min at cap" if effective_rpm > 0 else ""
    print(
        f"[{city_cfg.name}] {total} points remaining, realtime mode, "
        f"{workers} workers, {max_rpm} RPM, "
        f"{max_tpm or 'no'} TPM{eta_msg}…"
    )

    rpm_limiter = RateLimiter(max_rpm, 1.0)
    tpm_limiter = RateLimiter(max_tpm, float(estimated_tokens_per_request))
    ensure_dirs(city_cfg.out_jsonl, city_cfg.out_csv)
    csv_exists_ref = [city_cfg.out_csv.exists()]

    with (
        city_cfg.out_jsonl.open("a", encoding="utf-8") as jf,
        city_cfg.out_csv.open("a", encoding="utf-8", newline="") as cf,
    ):
        cw = csv.writer(cf)
        if not csv_exists_ref[0]:
            cw.writerow(["image", "answer"])
            csv_exists_ref[0] = True

        write_lock = threading.Lock()
        pbar = tqdm(total=total, desc=f"{city_cfg.name} captions", unit="pt")

        def worker(pid: int) -> dict:
            img = composite_bytes(city_cfg.street_view_dir, pid)
            rpm_limiter.wait_turn()
            tpm_limiter.wait_turn()
            res = call_openai_realtime(
                api_key=api_key, model=model,
                system_prompt=system_prompt, user_prompt=user_prompt,
                image_bytes=img, timeout_s=timeout_s,
                max_attempts=max_attempts, detail=detail,
            )
            cleaned, canonical = validate_and_canonicalize(res.parsed)
            return {
                "id": pid, "image": f"{pid}_combined.jpg",
                "json": cleaned, "answer": canonical,
                "model": model, "usage": res.usage,
                "ts": datetime.now(timezone.utc).isoformat(),
            }

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(worker, pid): pid for pid in todo}
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    rec = fut.result()
                except FileNotFoundError:
                    rec = {
                        "id": pid, "image": f"{pid}_combined.jpg",
                        "error": "missing heading image(s)",
                        "model": model, "ts": datetime.now(timezone.utc).isoformat(),
                    }
                except Exception as exc:
                    rec = {
                        "id": pid, "image": f"{pid}_combined.jpg",
                        "error": str(exc), "model": model,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                with write_lock:
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jf.flush()
                    if "answer" in rec:
                        cw.writerow([rec["image"], rec["answer"]])
                        cf.flush()
                pbar.update(1)
        pbar.close()


# ---------------------------------------------------------------------------
# Cost reporting
# ---------------------------------------------------------------------------

_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4-nano":  (0.20,  1.25),   # DEFAULT — newest, best for classification
    "gpt-4o-mini":   (0.15,  0.60),
    "gpt-5.4-mini":  (0.75,  4.50),
    "gpt-5.4":       (2.50, 15.00),
    "gpt-4o":        (2.50, 10.00),
    "gpt-4.1-nano":  (0.10,  0.40),
}


def print_cost_from_jsonl(jsonl_path: Path, *, model: str, batch: bool) -> None:
    if not jsonl_path.exists():
        print(f"(cost) JSONL not found: {jsonl_path}")
        return
    prompt_tok = out_tok = have = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            usage = rec.get("usage")
            if not isinstance(usage, dict):
                continue
            pt = usage.get("prompt_tokens") or 0
            ct = usage.get("completion_tokens") or 0
            if pt or ct:
                prompt_tok += int(pt)
                out_tok += int(ct)
                have += 1

    base = _PRICING.get(model, (2.50, 10.00))
    mult = 0.5 if batch else 1.0
    inp_price = base[0] * mult
    out_price = base[1] * mult
    cost_in  = (prompt_tok / 1_000_000) * inp_price
    cost_out = (out_tok    / 1_000_000) * out_price
    print(f"(cost) {have} records from {jsonl_path.name}")
    print(f"(cost) prompt tokens: {prompt_tok:,}  completion tokens: {out_tok:,}")
    print(f"(cost) model={model}  batch={batch}")
    print(f"(cost) estimated cost: ${cost_in + cost_out:.4f}")


def print_budget_estimate(
    cities: list[str],
    city_cfgs: dict[str, CityConfig],
    model: str,
    batch: bool,
    requested_ids: set[int] | None = None,
) -> None:
    INPUT_TOKENS_PER_CALL = 2_200   # enriched prompt ~700 tok + image high-detail ~1500
    OUTPUT_TOKENS_PER_CALL = 150
    base = _PRICING.get(model, (2.50, 10.00))
    mult = 0.5 if batch else 1.0
    inp_price = base[0] * mult
    out_price = base[1] * mult

    print("\n--- Budget estimate (pre-run) ---")
    total_pts = 0
    for city_name in cities:
        cfg = city_cfgs[city_name]
        pts = 0
        if cfg.street_view_dir.exists():
            ids = discover_ids(cfg.street_view_dir)
            ids = restrict_ids(ids, requested_ids, city_name=city_name, source_label="current imagery")
            pts = len(ids)
        cost = (pts * INPUT_TOKENS_PER_CALL / 1_000_000 * inp_price
                + pts * OUTPUT_TOKENS_PER_CALL / 1_000_000 * out_price)
        print(f"  {city_name:8s}  {pts:>5} points  →  ~${cost:.2f}")
        total_pts += pts

    total_cost = (total_pts * INPUT_TOKENS_PER_CALL / 1_000_000 * inp_price
                  + total_pts * OUTPUT_TOKENS_PER_CALL / 1_000_000 * out_price)
    mode_label = "batch (-50%)" if batch else "standard"
    print(f"  TOTAL     {total_pts:>5} points  →  ~${total_cost:.2f}  [{model}, {mode_label}]")
    print(f"  Token assumptions: {INPUT_TOKENS_PER_CALL} input + {OUTPUT_TOKENS_PER_CALL} output per composite")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate structured OpenAI walkability captions for WalkCLIP v2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--city", choices=["msp", "seattle", "dc", "both", "all"], default="both")
    ap.add_argument(
        "--mode", choices=["realtime", "batch"], default="realtime",
        help="'realtime' = immediate Chat Completions. 'batch' = 50%% cheaper async."
    )
    ap.add_argument(
        "--model", default="gpt-5.4-nano",
        help="OpenAI model. Default: gpt-5.4-nano (recommended). Fallback: gpt-4o-mini."
    )
    ap.add_argument(
        "--detail", choices=["low", "high"], default="high",
        help="Image detail level. 'high' recommended for street scenes. Default: high."
    )
    ap.add_argument("--limit", type=int, default=0, help="Process only first N points per city (0=all).")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent workers for realtime mode.")
    ap.add_argument("--max-rpm", type=float, default=500.0, help="Rate limit req/min for realtime mode.")
    ap.add_argument(
        "--max-tpm",
        type=float,
        default=0.0,
        help="Optional token cap per minute for realtime mode (0 disables TPM throttling).",
    )
    ap.add_argument(
        "--estimated-tokens-per-request",
        type=int,
        default=4250,
        help="Estimated total tokens per request for realtime TPM throttling. Default matches existing gpt-4o-mini low-detail runs.",
    )
    ap.add_argument("--timeout-s", type=float, default=90.0)
    ap.add_argument("--max-attempts", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true", help="Re-generate all, ignoring existing JSONL.")
    ap.add_argument("--dry-run", action="store_true", help="Count IDs without calling the API.")
    ap.add_argument("--print-cost", action="store_true", help="Print cost from existing JSONL, no API calls.")
    ap.add_argument("--poll-interval", type=int, default=60, help="Batch poll interval seconds.")
    ap.add_argument(
        "--point-ids-file",
        type=Path,
        default=None,
        help="Optional text file with one numeric point_id per line. Only those IDs will be processed.",
    )
    ap.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix appended to output filenames before the extension, e.g. _final_lock.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    load_dotenv_if_present(root / ".env")

    city_cfgs = build_city_configs(root)
    if args.city in ("both", "all"):
        cities = ["msp", "seattle", "dc"]
    else:
        cities = [args.city]
    requested_ids = read_requested_ids(args.point_ids_file)
    if args.output_suffix:
        city_cfgs = {
            name: with_output_suffix(cfg, args.output_suffix)
            for name, cfg in city_cfgs.items()
        }

    if args.print_cost:
        for city_name in cities:
            print_cost_from_jsonl(
                city_cfgs[city_name].out_jsonl,
                model=args.model,
                batch=(args.mode == "batch"),
            )
        return 0

    print_budget_estimate(
        cities,
        city_cfgs,
        model=args.model,
        batch=(args.mode == "batch"),
        requested_ids=requested_ids,
    )

    if args.dry_run:
        for city_name in cities:
            cfg = city_cfgs[city_name]
            if not cfg.street_view_dir.exists():
                print(f"[{city_name}] Dir not found: {cfg.street_view_dir}")
                continue
            ids = discover_ids(cfg.street_view_dir)
            ids = restrict_ids(ids, requested_ids, city_name=city_name, source_label="current imagery")
            if args.limit:
                ids = ids[: args.limit]
            done = read_done_ids(cfg.out_jsonl)
            todo = [pid for pid in ids if pid not in done]
            print(
                f"[{city_name}] complete IDs: {len(ids)}  done: {len(done)}  todo: {len(todo)}"
            )
        return 0

    api_key = get_openai_api_key()
    system_prompt = SYSTEM_PROMPT
    user_prompt = build_prompt()

    for city_name in cities:
        cfg = city_cfgs[city_name]
        if not cfg.street_view_dir.exists():
            print(f"[{city_name}] Street view dir not found, skipping: {cfg.street_view_dir}")
            continue

        ids = discover_ids(cfg.street_view_dir)
        ids = restrict_ids(ids, requested_ids, city_name=city_name, source_label="current imagery")
        if args.limit:
            ids = ids[: args.limit]
        done = set() if args.overwrite else read_done_ids(cfg.out_jsonl)
        print(f"[{city_name}] {len(ids)} IDs found, {len(done)} already done.")

        if args.mode == "batch":
            run_batch_mode(
                api_key=api_key, city_cfg=cfg, model=args.model,
                system_prompt=system_prompt, user_prompt=user_prompt,
                ids=ids, done=done, poll_interval_s=args.poll_interval, detail=args.detail,
            )
        else:
            run_realtime_mode(
                api_key=api_key, city_cfg=cfg, model=args.model,
                system_prompt=system_prompt, user_prompt=user_prompt,
                ids=ids, done=done, workers=args.workers,
                max_rpm=args.max_rpm, max_tpm=args.max_tpm,
                estimated_tokens_per_request=args.estimated_tokens_per_request,
                timeout_s=args.timeout_s,
                max_attempts=args.max_attempts, detail=args.detail,
            )

        print(f"[{city_name}] Done.\n  JSONL: {cfg.out_jsonl}\n  CSV:   {cfg.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())