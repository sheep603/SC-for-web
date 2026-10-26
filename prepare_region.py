"""Prepare Web-search data for the same 3-D view used by GridWorld.

The primary coordinates are the theory-level quantities used in the paper:
  x = L       normalized semantic communication cost
  y = D_L     legitimate-receiver MAP distortion of W
  z = S_e     semantic secrecy advantage I(S;Y_L)-I(S;Z_E)

The script preserves application-level retrieval metrics as auxiliary columns
and emits a deterministic CSV plus a JSON metadata file.  It does not invent
an interpolated surface: for a finite codebook, the defensible representation
is the set of achievable points and its nondominated/Pareto subset.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THEORY = PROJECT_ROOT / "results" / "web_theory" / "MSMARCO_Bob更强_全部确定性编码.csv"
PARETO = PROJECT_ROOT / "results" / "web_theory" / "MSMARCO_Bob更强_Pareto边界.csv"
APP = PROJECT_ROOT / "results" / "web_search" / "MSMARCO_Web_Search_Pareto逐编码结果.csv"
OUT_DIR = PROJECT_ROOT / "results" / "web_region"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return row.get("encoding_id", "")


def as_float(row, field):
    value = row.get(field, "")
    return float(value) if value not in (None, "", "NA") else None


def prepare(theory_rows, pareto_rows, app_rows):
    app_by_id = {key(row): row for row in app_rows if row.get("encoding_id") not in (None, "", "NA")}
    pareto_ids = {key(row) for row in pareto_rows}
    output = []
    for row in theory_rows:
        encoding_id = key(row)
        app = app_by_id.get(encoding_id, {})
        output.append(
            {
                "encoding_id": encoding_id,
                "encoding": row.get("encoding", ""),
                "is_pareto": int(encoding_id in pareto_ids),
                "cost_L": as_float(row, "L"),
                "distortion_D_L": as_float(row, "D_B"),
                "distortion_D_E": as_float(row, "D_E"),
                "secrecy_advantage_S_e": as_float(row, "Se"),
                "mutual_information_I_L": as_float(row, "I_B"),
                "mutual_information_I_E": as_float(row, "I_E"),
                "queries": as_float(app, "queries"),
                "MRR_at_31": as_float(app, "MRR@31"),
                "Recall_at_1": as_float(app, "Recall@1"),
                "Recall_at_10": as_float(app, "Recall@10"),
                "NDCG_at_10": as_float(app, "NDCG@10"),
            }
        )
    return output


def main():
    for required in (THEORY, PARETO, APP):
        if not required.exists():
            raise FileNotFoundError(required)

    theory_rows = read_csv(THEORY)
    pareto_rows = read_csv(PARETO)
    app_rows = read_csv(APP)
    rows = prepare(theory_rows, pareto_rows, app_rows)
    pareto_rows_out = [row for row in rows if row["is_pareto"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fields = list(rows[0])
    with (OUT_DIR / "MSMARCO_Web_3D_all_6561_encodings.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT_DIR / "MSMARCO_Web_3D_pareto_242_points.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pareto_rows_out)

    paper_points = [
        ("Security-unaware SC", "1.5", "2314"),
        ("Proposed SC", "1.5", "2308"),
        ("Security-unaware SC", "2.0", "4856"),
        ("Proposed SC", "2.0", "1345"),
        ("Security-unaware SC", "2.5", "5102"),
        ("Proposed SC", "2.5", "5722"),
    ]
    with (OUT_DIR / "table1_semantic_security.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Scheme", "Budget_B", "Encoding_ID"))
        writer.writerows(paper_points)

    metadata = {
        "title": "Web-search distortion-cost-secrecy achievable points",
        "coordinates": {
            "x": "cost_L = L",
            "y": "distortion_D_L = D_L, legitimate-receiver MAP distortion of finite Web information-need state W",
            "z": "secrecy_advantage_S_e = I(S;Y_L)-I(S;Z_E)",
        },
        "auxiliary_columns": ["distortion_D_E", "mutual_information_I_L", "mutual_information_I_E", "MRR_at_31", "Recall_at_1", "Recall_at_10", "NDCG_at_10"],
        "source_files": [str(THEORY), str(PARETO), str(APP)],
        "all_deterministic_encodings": len(rows),
        "pareto_points": len(pareto_rows_out),
        "recommendation": "Plot all 6561 points as a light 3-D cloud and overlay the 242 nondominated points. If a triangulated surface is needed for visualization, label it as an interpolation/visual guide, not as additional achievable points.",
        "web_task_scope": "MS MARCO-derived query-local ranking with one supplied positive and 30 supplied negatives per query; not official full-corpus MS MARCO evaluation.",
    }
    (OUT_DIR / "MSMARCO_Web_3D_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} all-code points and {len(pareto_rows_out)} Pareto points to {OUT_DIR}")


if __name__ == "__main__":
    main()
