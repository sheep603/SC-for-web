"""Run the isolated external-baseline comparison.

The run is deliberately deterministic and produces data from the implemented
adapters. It does not edit any earlier experiment directory. The selected
external operating point for each common budget is chosen on validation
queries by retrieval utility subject to the validation cost budget, then
evaluated once on the held-out test split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baseline_models import (
    REFINE_FRACTIONS,
    SEED,
    TARGET_BUDGETS,
    Context,
    build_context,
    evaluate_policy_on_split,
    information_density,
    make_c2f_packets,
    make_plm_packets,
    make_vbr_packets,
    paper_eval,
    paper_reference_candidates,
    plm_embeddings,
    plm_intent_predictions,
    query_uncertainty,
    save_json,
    threshold_for_fraction,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "results" / "external_baselines"


def model_candidates(ctx: Context):
    """Build validation-calibrated packet policies for all three mechanisms."""
    train_uncertainty = query_uncertainty(ctx.train, ctx.centroids)
    validation_uncertainty = query_uncertainty(ctx.validation, ctx.centroids)
    test_uncertainty = query_uncertainty(ctx.test, ctx.centroids)

    train_density = information_density(ctx.train, ctx.idf)
    validation_density = information_density(ctx.validation, ctx.idf)
    test_density = information_density(ctx.test, ctx.idf)

    (plm_train, plm_validation, plm_test), plm_backend = plm_embeddings(
        ctx.train, [ctx.validation, ctx.test]
    )
    (plm_validation_labels, plm_validation_uncertainty), (plm_test_labels, plm_test_uncertainty) = plm_intent_predictions(
        plm_train, [plm_validation, plm_test], ctx.train_labels
    )
    plm_train_labels = ctx.train_labels
    # The training uncertainty is computed against the PLM prototypes through
    # the same prediction path used by validation and test.
    plm_train_uncertainty = plm_intent_predictions(
        plm_train, [plm_train], ctx.train_labels
    )[0][1]

    output = []
    for fraction in REFINE_FRACTIONS:
        threshold = threshold_for_fraction(validation_uncertainty, fraction)
        output.append(
            {
                "model": "Coarse-to-Fine TVT 2025",
                "model_key": "c2f_tvt_2025",
                "mechanism": "group packet plus optional intent refinement",
                "backend": "hashed_query_embedding",
                "policy_id": f"refine_fraction_{fraction:.2f}",
                "threshold": threshold,
                "train_packets": make_c2f_packets(ctx.train_labels, train_uncertainty, threshold),
                "validation_packets": make_c2f_packets(ctx.validation_labels, validation_uncertainty, threshold),
                "test_packets": make_c2f_packets(ctx.test_labels, test_uncertainty, threshold),
            }
        )
        threshold = threshold_for_fraction(validation_density, fraction)
        output.append(
            {
                "model": "Two-Phase VBR DCN 2026",
                "model_key": "vbr_dcn_2026",
                "mechanism": "information-density controller with optional fine packet",
                "backend": "query_length_plus_training_idf",
                "policy_id": f"refine_fraction_{fraction:.2f}",
                "threshold": threshold,
                "train_packets": make_vbr_packets(ctx.train_labels, train_density, threshold),
                "validation_packets": make_vbr_packets(ctx.validation_labels, validation_density, threshold),
                "test_packets": make_vbr_packets(ctx.test_labels, test_density, threshold),
            }
        )
        threshold = threshold_for_fraction(plm_validation_uncertainty, fraction)
        output.append(
            {
                "model": "PLM Two-Stage TCCN 2026",
                "model_key": "plm_tccn_2026",
                "mechanism": "query encoder with group packet plus intent refinement",
                "backend": plm_backend,
                "policy_id": f"refine_fraction_{fraction:.2f}",
                "threshold": threshold,
                "train_packets": make_plm_packets(plm_train_labels, plm_train_uncertainty, threshold),
                "validation_packets": make_plm_packets(plm_validation_labels, plm_validation_uncertainty, threshold),
                "test_packets": make_plm_packets(plm_test_labels, plm_test_uncertainty, threshold),
            }
        )
    return output, plm_backend


def choose_validation_policy(ctx: Context, candidates: list[dict]):
    selected = []
    diagnostics = []
    for candidate in candidates:
        summary, _ = evaluate_policy_on_split(
            ctx,
            candidate["train_packets"],
            ctx.validation,
            ctx.validation_labels,
            candidate["validation_packets"],
        )
        candidate["validation_summary"] = summary
        for budget in TARGET_BUDGETS:
            if summary["L_common"] <= budget + 1e-12:
                diagnostics.append(
                    {
                        "model": candidate["model"],
                        "model_key": candidate["model_key"],
                        "backend": candidate["backend"],
                        "policy_id": candidate["policy_id"],
                        "budget": budget,
                        "validation_L_common": summary["L_common"],
                        "validation_D_L": summary["D_L"],
                        "validation_S_e": summary["S_e"],
                        "validation_MRR@31": summary["MRR@31"],
                        "validation_Recall@10": summary["Recall@10"],
                    }
                )
    for model_key in sorted({candidate["model_key"] for candidate in candidates}):
        model_candidates_for_key = [candidate for candidate in candidates if candidate["model_key"] == model_key]
        for budget in TARGET_BUDGETS:
            feasible = [candidate for candidate in model_candidates_for_key if candidate["validation_summary"]["L_common"] <= budget + 1e-12]
            if not feasible:
                continue
            # Retrieval utility is the primary task objective for these
            # security-unaware external baselines. Cost is the hard constraint;
            # lower distortion and lower cost break ties.
            best = max(
                feasible,
                key=lambda candidate: (
                    candidate["validation_summary"]["MRR@31"],
                    candidate["validation_summary"]["Recall@10"],
                    -candidate["validation_summary"]["D_L"],
                    -candidate["validation_summary"]["L_common"],
                ),
            )
            selected.append((best, budget))
    return selected, diagnostics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = build_context()
    candidates, plm_backend = model_candidates(ctx)
    selected, diagnostics = choose_validation_policy(ctx, candidates)
    write_csv(OUT / "validation_policy_selection.csv", diagnostics)

    summary_rows = []
    per_query_rows = []
    seen_selected = set()
    for candidate, budget in selected:
        summary, per_query = evaluate_policy_on_split(
            ctx,
            candidate["train_packets"],
            ctx.test,
            ctx.test_labels,
            candidate["test_packets"],
        )
        key = (candidate["model_key"], budget)
        seen_selected.add(key)
        row = {
            "family": "external_baseline",
            "model": candidate["model"],
            "model_key": candidate["model_key"],
            "backend": candidate["backend"],
            "mechanism": candidate["mechanism"],
            "budget": budget,
            "policy_id": candidate["policy_id"],
            "threshold": candidate["threshold"],
            "queries": len(ctx.test),
            "design_L": float(candidate["validation_summary"]["L_common"]),
            **summary,
        }
        summary_rows.append(row)
        for item in per_query:
            per_query_rows.append(
                {
                    "family": "external_baseline",
                    "model": candidate["model"],
                    "model_key": candidate["model_key"],
                    "backend": candidate["backend"],
                    "budget": budget,
                    **item,
                }
            )

    # Evaluate the two existing finite-code choices through the same packet
    # observation evaluator so the comparison uses one metric implementation.
    for scheme, budget, metric in paper_reference_candidates(ctx):
        encoding = tuple(metric["encoding_tuple"])
        summary, per_query = paper_eval(ctx, ctx.train_labels, ctx.test_labels, ctx.test, encoding)
        summary_rows.append(
            {
                "family": "paper_reference",
                "model": scheme,
                "model_key": "security_unaware_finite" if scheme.startswith("Security") else "proposed_secrecy_aware_finite",
                "backend": "existing_finite_codebook",
                "mechanism": "existing paper finite-code operating point",
                "budget": budget,
                "policy_id": f"encoding_{metric['encoding_id']}",
                "threshold": "NA",
                "queries": len(ctx.test),
                "design_L": float(metric["L"]),
                **summary,
                "paper_cost_L": float(summary.get("paper_cost_L", summary["L_common"])),
            }
        )
        for item in per_query:
            per_query_rows.append(
                {
                    "family": "paper_reference",
                    "model": scheme,
                    "model_key": "security_unaware_finite" if scheme.startswith("Security") else "proposed_secrecy_aware_finite",
                    "backend": "existing_finite_codebook",
                    "budget": budget,
                    **item,
                }
            )

    write_csv(OUT / "external_baselines_summary.csv", summary_rows)
    write_csv(OUT / "external_baselines_per_query.csv", per_query_rows)

    proposed = {
        row["budget"]: row
        for row in summary_rows
        if row["model_key"] == "proposed_secrecy_aware_finite"
    }
    comparison = []
    for row in summary_rows:
        if row["family"] != "external_baseline" or row["budget"] not in proposed:
            continue
        reference = proposed[row["budget"]]
        comparison.append(
            {
                "budget": row["budget"],
                "external_model": row["model"],
                "external_policy": row["policy_id"],
                "proposed_model": reference["model"],
                "delta_external_minus_proposed_L": row["L_common"] - reference["L_common"],
                "delta_external_minus_proposed_D_L": row["D_L"] - reference["D_L"],
                "delta_external_minus_proposed_S_e": row["S_e"] - reference["S_e"],
                "delta_external_minus_proposed_MRR@31": row["MRR@31"] - reference["MRR@31"],
                "delta_external_minus_proposed_Recall@10": row["Recall@10"] - reference["Recall@10"],
                "delta_external_minus_proposed_NDCG@10": row["NDCG@10"] - reference["NDCG@10"],
                "external_I_W_E": row["I_W_E"],
                "proposed_I_W_E": reference["I_W_E"],
            }
        )
    write_csv(OUT / "external_vs_paper.csv", comparison)

    audit = {
        "seed": SEED,
        "records_scanned": ctx.scanned,
        "malformed_records": ctx.malformed,
        "split": {"train": len(ctx.train), "validation": len(ctx.validation), "test": len(ctx.test)},
        "candidate_pool": "one positive plus 30 supplied negatives; 31-candidate query-local ranking",
        "retriever": "BM25(k1=1.2,b=0.75), training-only IDF and passage statistics",
        "source": "query-only; passages and relevance labels are evaluation-only for external policy selection",
        "alphabet_size": 13,
        "bits_per_symbol": 4,
        "channel": "existing Bob-better mode-conditioned DMC; independent channel use per packet position",
        "packet_length_observable": True,
        "selection": "validation retrieval MRR@31 under validation L_common budget; test labels not used for policy selection",
        "budgets": TARGET_BUDGETS,
        "external_models": [
            "Coarse-to-Fine TVT 2025",
            "Two-Phase VBR DCN 2026",
            "PLM Two-Stage TCCN 2026",
        ],
        "plm_backend": plm_backend,
        "selected_points": len(selected),
        "paper_reference_points": sum(1 for row in summary_rows if row["family"] == "paper_reference"),
        "files": [
            "external_baselines_summary.csv",
            "external_baselines_per_query.csv",
            "external_vs_paper.csv",
            "validation_policy_selection.csv",
        ],
        "interpretation": "External rows are mechanism-level Web adapters. PLM rows are not published-checkpoint results unless WEB_PLM_MODEL_PATH points to the corresponding local checkpoint.",
    }
    save_json(OUT / "experiment_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    for row in summary_rows:
        print(row["family"], row["model"], row["budget"], row["L_common"], row["D_L"], row["S_e"], row["MRR@31"])


if __name__ == "__main__":
    main()
