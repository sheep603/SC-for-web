"""Three-way Web Search baseline comparison on the query-only MS MARCO model.

The script compares raw-query transmission, distortion-oriented semantic
communication, and confidentiality-aware semantic communication without
changing the MS MARCO split, fixed 31-passage pools, BM25 statistics, or the
legitimate/eavesdropper degraded-channel parameters.

Raw queries are not members of the 13-symbol semantic alphabet.  To avoid
pretending otherwise, they use an alphabet extension of the same fine-mode
symmetric DMC: each complete raw query is one raw-message codeword, Q_L has
the same correct-codeword probability (0.96), and Q_E=Q_L T uses the same
fine-mode Eve retention (0.65).  Its payload cost is the number of lexical
tokens; semantic costs retain the theory's normalized 1/2/3 code-length units.
"""
import csv
import math
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from secrecy_region import (
    COST,
    K,
    N_S,
    SEED,
    build_channels,
    exact_metrics,
    fit_intents,
    load_data,
    pareto,
    symbol,
)
from web_search import (
    N_NEG,
    bm25_score,
    build_bm25_index,
    expected_retrieval,
    fit_query_embedding,
    mean_passage_length,
    precompute_output_metrics,
    rank_metrics,
    representative_queries,
    tokens,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "msmarco"
OUT = PROJECT_ROOT / "results" / "web_ablation_baselines"
RAW_ERROR_SAMPLES = 64
RAW_EVE_RETAIN_FINE = 0.65
RAW_MODE_CORRECT_PROBABILITY = 0.96
BUDGETS = (2.0, 2.5)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mutual_information_from_joint(joint):
    """Return I(W;Y) for a finite normalized joint distribution."""
    p_w = joint.sum(axis=1, keepdims=True)
    p_y = joint.sum(axis=0, keepdims=True)
    mask = joint > 0.0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / (p_w * p_y)[mask])))


def state_metrics(encoding, prior_w, channel_l, channel_e):
    """Evaluate common W-state metrics for a selected semantic encoder."""
    def joint(channel):
        out = np.zeros((K, N_S))
        for w, mode in enumerate(encoding):
            out[w] = prior_w[w] * channel[symbol(w, mode)]
        return out

    jl, je = joint(channel_l), joint(channel_e)
    i_l, i_e = mutual_information_from_joint(jl), mutual_information_from_joint(je)
    return {
        "D_L": float(1.0 - jl.max(axis=0).sum()),
        "D_E": float(1.0 - je.max(axis=0).sum()),
        "I_W_L": i_l,
        "I_W_E": i_e,
        "S_e_W": i_l - i_e,
        "communication_cost": float(sum(prior_w[w] * COST[mode] for w, mode in enumerate(encoding))),
    }


def raw_state_metrics(labels):
    """Exact W-state metrics for the raw-query alphabet extension.

    The input distribution is uniform over the held-out raw query codewords.
    Q_L is the fine-mode N-ary symmetric DMC.  Q_E is its physically degraded
    post-channel, using the fine-mode retention parameter from the main model.
    """
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    counts = np.bincount(labels, minlength=K).astype(float)

    def joint_from_correct_probability(p_correct):
        p_wrong = (1.0 - p_correct) / (n - 1)
        joint = np.empty((K, n))
        for output_index, w in enumerate(labels):
            joint[:, output_index] = p_wrong * counts / n
            joint[w, output_index] += (p_correct - p_wrong) / n
        return joint

    p_l = RAW_MODE_CORRECT_PROBABILITY
    # Q_E = alpha Q_L + (1-alpha) U for the N-ary extension of T.
    p_e = RAW_EVE_RETAIN_FINE * p_l + (1.0 - RAW_EVE_RETAIN_FINE) / n
    jl, je = joint_from_correct_probability(p_l), joint_from_correct_probability(p_e)
    i_l, i_e = mutual_information_from_joint(jl), mutual_information_from_joint(je)
    return {
        "D_L": float(1.0 - jl.max(axis=0).sum()),
        "D_E": float(1.0 - je.max(axis=0).sum()),
        "I_W_L": i_l,
        "I_W_E": i_e,
        "S_e_W": i_l - i_e,
        "raw_channel_correct_probability": p_l,
        "raw_eve_retention": RAW_EVE_RETAIN_FINE,
    }


def raw_retrieval_metrics(test, idf, avgdl):
    """Estimate raw-channel Web Search utility with fixed error-query samples.

    For a correct raw-message observation, the receiver uses the original
    query.  For an erroneous observation, the N-ary symmetric DMC emits a
    uniformly selected different held-out raw query.  The error expectation is
    approximated by a fixed-seed sample; its finite sample size is reported.
    """
    rng = np.random.default_rng(SEED + 917)
    n = len(test)
    totals = Counter()
    per_query = {key: [] for key in rank_metrics(1)}
    p_correct = RAW_MODE_CORRECT_PROBABILITY
    for i, item in enumerate(test):
        candidates = [item["positive"]] + item["negative"][:N_NEG]

        def metrics_for_query(query):
            query_terms = tokens(query)
            scores = [bm25_score(query_terms, passage, idf, avgdl) for passage in candidates]
            order = sorted(range(len(candidates)), key=lambda j: (-scores[j], j))
            return rank_metrics(order.index(0) + 1)

        correct = metrics_for_query(item["query"])
        draws = rng.integers(0, n - 1, size=RAW_ERROR_SAMPLES)
        wrong_indices = draws + (draws >= i)
        wrong_totals = Counter()
        for j in wrong_indices:
            for key, value in metrics_for_query(test[int(j)]["query"]).items():
                wrong_totals[key] += value
        for key in correct:
            expected = p_correct * correct[key] + (1.0 - p_correct) * wrong_totals[key] / RAW_ERROR_SAMPLES
            totals[key] += expected
            per_query[key].append(expected)
    result = {key: totals[key] / n for key in totals}
    result.update({f"{key}_query_SE": float(np.std(values, ddof=1) / math.sqrt(n)) for key, values in per_query.items()})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, scanned, malformed = load_data()
    train, validation, test = data[:8400], data[8400:10200], data[10200:]
    _, centroids, train_labels, train_prior = fit_intents(train)
    test_vectors = np.vstack([fit_query_embedding(item) for item in test])
    test_labels = ((test_vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(2).argmin(1)
    test_prior = np.bincount(test_labels, minlength=K).astype(float)
    test_prior /= test_prior.sum()

    # Selection is training-only.  Evaluation uses the same held-out queries for all methods.
    q_l, q_e = build_channels("Bob-better")[:2]
    theory_rows = []
    for encoding_id, encoding in enumerate(product(range(3), repeat=K)):
        train_metric = exact_metrics(encoding, train_prior, q_l, q_e)
        train_metric.update({"encoding_id": encoding_id, "encoding_tuple": encoding})
        theory_rows.append(train_metric)
    boundary_ids = {row["encoding_id"] for row in pareto(theory_rows)}

    idf = build_bm25_index(train)
    avgdl = mean_passage_length(train)
    _, message_queries = representative_queries(train, centroids, train_labels, train_prior)
    cached_semantic = precompute_output_metrics(test, message_queries, idf, avgdl)

    raw_metrics = raw_state_metrics(test_labels)
    raw_metrics["communication_cost"] = float(np.mean([len(tokens(item["query"])) for item in test]))
    raw_retrieval = raw_retrieval_metrics(test, idf, avgdl)
    rows = [{
        "budget": "not constrained",
        "scheme": "Raw Query Transmission",
        "selection_objective": "Transmit the original held-out query without semantic compression",
        "encoding_id": "NA",
        "encoding": "NA",
        "cost_unit": "normalized lexical-token-equivalent payload units",
        "communication_cost": raw_metrics["communication_cost"],
        "D_L": raw_metrics["D_L"],
        "D_E": raw_metrics["D_E"],
        "I_W_L": raw_metrics["I_W_L"],
        "I_W_E": raw_metrics["I_W_E"],
        "S_e_W": raw_metrics["S_e_W"],
        "I_S_L": "NA",
        "I_S_E": "NA",
        "S_e_message": "NA",
        "queries": len(test),
        "raw_error_samples": RAW_ERROR_SAMPLES,
        **raw_retrieval,
    }]

    for budget in BUDGETS:
        feasible = [row for row in theory_rows if row["L"] <= budget + 1e-12]
        selected = [
            ("Security-unaware Semantic Communication", "Minimize D_L subject to training L<=B", min(feasible, key=lambda row: (row["D_B"], row["L"], -row["Se"]))),
            ("Proposed Confidentiality-aware Semantic Communication", "Maximize S_e subject to training L<=B", max(feasible, key=lambda row: (row["Se"], row["D_B"], row["L"]))),
        ]
        for scheme, objective, selected_row in selected:
            encoding = selected_row["encoding_tuple"]
            eval_state = state_metrics(encoding, test_prior, q_l, q_e)
            retrieval = expected_retrieval(cached_semantic, test_labels, encoding, q_l)
            rows.append({
                "budget": budget,
                "scheme": scheme,
                "selection_objective": objective,
                "encoding_id": selected_row["encoding_id"],
                "encoding": " ".join(map(str, encoding)),
                "cost_unit": "normalized lexical-token-equivalent payload units",
                "communication_cost": eval_state["communication_cost"],
                "D_L": eval_state["D_L"],
                "D_E": eval_state["D_E"],
                "I_W_L": eval_state["I_W_L"],
                "I_W_E": eval_state["I_W_E"],
                "S_e_W": eval_state["S_e_W"],
                "I_S_L": selected_row["I_B"],
                "I_S_E": selected_row["I_E"],
                "S_e_message": selected_row["Se"],
                "queries": len(test),
                "raw_error_samples": "NA",
                **retrieval,
            })

    write_csv(OUT / "MSMARCO_三基线_性能对比.csv", rows)
    pairwise = []
    raw = rows[0]
    for budget in BUDGETS:
        unaware = next(row for row in rows if row["budget"] == budget and row["scheme"].startswith("Security-unaware"))
        proposed = next(row for row in rows if row["budget"] == budget and row["scheme"].startswith("Proposed"))
        for reference, target, question in [
            (raw, unaware, "semantic communication versus raw query transmission"),
            (unaware, proposed, "confidentiality-aware versus security-unaware semantic communication"),
        ]:
            pairwise.append({
                "budget": budget,
                "scientific_question": question,
                "reference": reference["scheme"],
                "target": target["scheme"],
                "target_minus_reference_cost": float(target["communication_cost"]) - float(reference["communication_cost"]),
                "target_minus_reference_D_L": float(target["D_L"]) - float(reference["D_L"]),
                "target_minus_reference_S_e_W": float(target["S_e_W"]) - float(reference["S_e_W"]),
                "target_minus_reference_MRR4": float(target["MRR@4"]) - float(reference["MRR@4"]),
                "target_minus_reference_MRR31": float(target["MRR@31"]) - float(reference["MRR@31"]),
                "target_minus_reference_Recall3": float(target["Recall@3"]) - float(reference["Recall@3"]),
                "target_minus_reference_Recall10": float(target["Recall@10"]) - float(reference["Recall@10"]),
                "target_minus_reference_NDCG3": float(target["NDCG@3"]) - float(reference["NDCG@3"]),
                "target_minus_reference_NDCG10": float(target["NDCG@10"]) - float(reference["NDCG@10"]),
            })
    write_csv(OUT / "MSMARCO_三基线_成对差异.csv", pairwise)

    with open(OUT / "MSMARCO_三基线_实验审计.txt", "w", encoding="utf-8") as handle:
        handle.write("MS MARCO query-only three-baseline comparison\n")
        handle.write(f"records_scanned={scanned}; malformed={malformed}; sample={len(data)}\n")
        handle.write(f"split=train={len(train)}, validation={len(validation)}, test={len(test)}\n")
        handle.write("All methods use the same held-out test queries, one positive plus 30 negatives per query, training-only BM25 IDF, and BM25(k1=1.2,b=0.75).\n")
        handle.write("Semantic W/prototypes/prior/representative queries use query text only. Passages and relevance labels are retrieval-evaluation only.\n")
        handle.write("Semantic methods use the existing 13-symbol Q_L/Q_E model. Raw query transmission extends the identical fine-mode symmetric DMC and Eve post-channel to the raw-query codeword alphabet: p_L=0.96, alpha_E=0.65.\n")
        handle.write("D_L, D_E, I_W_L, I_W_E, and S_e_W=I(W;Y_L)-I(W;Z_E) are common held-out semantic-state metrics. Semantic S_e_message=I(S;Y_L)-I(S;Z_E) is retained only as a method-specific auxiliary quantity.\n")
        handle.write("Communication cost is reported in a declared common payload scale: raw cost is the lexical-token count/query and one original semantic code-length unit (1/2/3) is treated as one lexical-token-equivalent payload unit. This is a normalized payload comparison, not measured bytes, latency, or physical channel uses.\n")
        handle.write(f"Raw erroneous-query retrieval uses {RAW_ERROR_SAMPLES} fixed-seed symmetric-channel samples/query; semantic retrieval is evaluated exactly over Q_L outputs.\n")
        handle.write(f"training_encodings_enumerated={len(theory_rows)}; training_pareto_encodings={len(boundary_ids)}\n")
    print("OUTPUT", OUT)
    for row in rows:
        print(row["budget"], row["scheme"], "cost=", row["communication_cost"], "D_L=", row["D_L"], "S_e_W=", row["S_e_W"], "MRR@31=", row["MRR@31"])


if __name__ == "__main__":
    main()
