"""
Decision test: can frozen imagery predict crosswalks cross-city under a *better*
label (OSM marked crossings) than Overture?

Runs THREE binary labels through ONE identical frozen-siglip2 (street+NAIP) ->
PCA-128/modality -> StandardScaler -> LogisticRegression pipeline, under the LOCO
protocol (6 MSP/Seattle/DC directions + Pittsburgh zero-shot). Two labels are
calibration anchors with known frozen numbers:

    overture_sidewalk_present    ~0.76 AUROC  (upper anchor)
    overture_crosswalk_present   ~0.56 AUROC  (the dropped target's old number)
    osm_marked_crossing_present  = the test   (does a cleaner label unlock signal?)

Decision rule (printed): if osm_marked AUROC climbs well above the Overture-crosswalk
anchor and toward sidewalk, crosswalk earns a target slot. If it stays ~0.55 with a
cleaner label, crosswalk is a hard visual target -> stays killed.

Inputs (already present): siglip2 embeddings for all 4 cities; FLA;
osm_crossing_labels.csv (from fetch_osm_crossings.py).

Output:
    project/artifacts/reports/crosswalk_label_eval.json

Run (Windows, repo ROOT):
    & ".venv\\Scripts\\python.exe" project\\scripts\\eval_crosswalk_labels.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBED_DIR = REPO_ROOT / "project/artifacts/embeddings"
LABELS_CSV = REPO_ROOT / "project/data/processed/labels/features_labels_agreement.csv"
OSM_CSV = REPO_ROOT / "project/data/processed/labels/osm_crossing_labels.csv"
REPORT_OUT = REPO_ROOT / "project/artifacts/reports/crosswalk_label_eval.json"

BACKBONE = "siglip2"
TRIO = ["msp", "seattle", "dc"]
CITIES = ["msp", "seattle", "dc", "pittsburgh"]
PCA_DIM = 128
SEED = 42

LABELS = {
    "overture_sidewalk_present": "anchor_high (~0.76 known)",
    "overture_crosswalk_present": "anchor_low (~0.56 known, dropped target)",
    "osm_marked_crossing_present": "TEST (cleaner crosswalk label)",
}


def load_emb(source: str, city: str) -> tuple[np.ndarray, np.ndarray]:
    emb = np.load(EMBED_DIR / f"{BACKBONE}_{source}_{city}.npy").astype(np.float32)
    ids = np.load(EMBED_DIR / f"{BACKBONE}_{source}_{city}_ids.npy").astype(np.int64)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    return emb, ids


def city_block(city: str, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Aligned (street, naip, label_frame) for points present in embeddings AND labels."""
    se, sid = load_emb("street", city)
    ne, nid = load_emb("naip", city)
    smap = {int(p): i for i, p in enumerate(sid)}
    nmap = {int(p): i for i, p in enumerate(nid)}
    lab = labels[labels["city"] == city]
    s_rows, n_rows, keep = [], [], []
    for pid in lab["point_id"].astype(int):
        if pid in smap and pid in nmap:
            s_rows.append(se[smap[pid]])
            n_rows.append(ne[nmap[pid]])
            keep.append(pid)
    frame = lab.set_index("point_id").loc[keep].reset_index()
    return np.asarray(s_rows, dtype=np.float32), np.asarray(n_rows, dtype=np.float32), frame


def fit_eval(
    street_tr: np.ndarray, naip_tr: np.ndarray, y_tr: np.ndarray,
    street_te: np.ndarray, naip_te: np.ndarray, y_te: np.ndarray,
) -> float | None:
    m_tr, m_te = np.isfinite(y_tr), np.isfinite(y_te)
    street_tr, naip_tr, y_tr = street_tr[m_tr], naip_tr[m_tr], y_tr[m_tr].astype(int)
    street_te, naip_te, y_te = street_te[m_te], naip_te[m_te], y_te[m_te].astype(int)
    if len(y_te) < 30 or len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
        return None
    parts_tr, parts_te = [], []
    for tr, te in ((street_tr, street_te), (naip_tr, naip_te)):
        k = min(PCA_DIM, tr.shape[1], tr.shape[0] - 1)
        pca = PCA(n_components=k, random_state=SEED).fit(tr)
        parts_tr.append(pca.transform(tr))
        parts_te.append(pca.transform(te))
    X_tr = np.concatenate(parts_tr, axis=1)
    X_te = np.concatenate(parts_te, axis=1)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    argparse.ArgumentParser(description="Crosswalk label decision test.").parse_args()

    fla = pd.read_csv(LABELS_CSV, usecols=["point_id", "city", "overture_sidewalk_present", "overture_crosswalk_present"])
    fla["point_id"] = fla["point_id"].astype(int)
    osm = pd.read_csv(OSM_CSV)
    osm["point_id"] = osm["point_id"].astype(int)
    labels = fla.merge(osm[["point_id", "city", "osm_marked_crossing_present"]], on=["point_id", "city"], how="left")

    blocks = {c: city_block(c, labels) for c in CITIES}
    for c in CITIES:
        logging.info("Loaded %s: %d aligned points", c, len(blocks[c][2]))

    report: dict = {"backbone": BACKBONE, "pca_dim": PCA_DIM, "by_label": {}}
    for label, role in LABELS.items():
        directions: dict[str, float] = {}
        for tr_city in TRIO:
            for te_city in TRIO:
                if tr_city == te_city:
                    continue
                s_tr, n_tr, f_tr = blocks[tr_city]
                s_te, n_te, f_te = blocks[te_city]
                auroc = fit_eval(s_tr, n_tr, f_tr[label].to_numpy(),
                                 s_te, n_te, f_te[label].to_numpy())
                if auroc is not None:
                    directions[f"{tr_city}->{te_city}"] = round(auroc, 4)
        # Pittsburgh zero-shot: train MSP+Seattle+DC -> test Pittsburgh
        s_tr = np.concatenate([blocks[c][0] for c in TRIO])
        n_tr = np.concatenate([blocks[c][1] for c in TRIO])
        y_tr = np.concatenate([blocks[c][2][label].to_numpy() for c in TRIO])
        s_pe, n_pe, f_pe = blocks["pittsburgh"]
        pgh = fit_eval(s_tr, n_tr, y_tr, s_pe, n_pe, f_pe[label].to_numpy())

        vals = list(directions.values())
        report["by_label"][label] = {
            "role": role,
            "loco_directions": directions,
            "loco_mean_auroc": round(float(np.mean(vals)), 4) if vals else None,
            "loco_std_auroc": round(float(np.std(vals)), 4) if vals else None,
            "pittsburgh_zeroshot_auroc": round(pgh, 4) if pgh is not None else None,
        }
        logging.info("%-30s LOCO mean AUROC=%s  Pittsburgh=%s",
                     label, report["by_label"][label]["loco_mean_auroc"],
                     report["by_label"][label]["pittsburgh_zeroshot_auroc"])

    # Verdict
    anchor = report["by_label"]["overture_crosswalk_present"]["loco_mean_auroc"]
    test = report["by_label"]["osm_marked_crossing_present"]["loco_mean_auroc"]
    sidewalk = report["by_label"]["overture_sidewalk_present"]["loco_mean_auroc"]
    gain = round(test - anchor, 4) if (test is not None and anchor is not None) else None
    if gain is not None and test >= 0.65 and gain >= 0.08:
        verdict = "REVIVE crosswalk: cleaner OSM label unlocks learnable cross-city signal."
    elif gain is not None and gain >= 0.05:
        verdict = "BORDERLINE: OSM label helps but stays below the visual targets -- judgment call."
    else:
        verdict = "KEEP KILLED: cleaner label does not unlock signal; crosswalk is a hard visual target."
    report["verdict"] = {
        "sidewalk_anchor": sidewalk, "overture_crosswalk_anchor": anchor,
        "osm_crosswalk_test": test, "gain_over_overture": gain, "decision": verdict,
    }

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("=" * 72)
    logging.info("sidewalk(anchor)=%s  overture-crosswalk(anchor)=%s  OSM-crosswalk(test)=%s  gain=%s",
                 sidewalk, anchor, test, gain)
    logging.info("VERDICT: %s", verdict)
    logging.info("Wrote: %s", REPORT_OUT)
    logging.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
