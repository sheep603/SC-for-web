"""Application-level Web Search validation for the finite MS MARCO model.

The script keeps the candidate passages and BM25 pipeline fixed.  It changes
only the finite semantic encoding and the legitimate receiver channel.  For
each received semantic symbol, a surrogate query is selected from training
queries only.  Retrieval metrics are averaged exactly over the discrete
channel output distribution, rather than estimated from one noise draw.
"""
import csv
import json
import math
import re
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
    mode_of,
    pareto,
    symbol,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "msmarco"
OUT = PROJECT_ROOT / "results" / "web_search"
FIG = OUT / "figures"
TOKEN = re.compile(r"[a-z0-9]+")
MAX_Q = 12000
N_NEG = 30


def tokens(text):
    return TOKEN.findall((text or "").lower())


def build_bm25_index(items):
    """Build a fixed BM25 IDF and passage statistics from training only."""
    df = Counter()
    n_docs = 0
    for item in items:
        passages = [item["positive"]] + item["negative"][:N_NEG]
        for passage in passages:
            terms = set(tokens((passage.get("title", "") + " " + passage.get("text", ""))))
            df.update(terms)
            n_docs += 1
    idf = {
        term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }
    return idf


def bm25_score(query_terms, passage, idf, avgdl, k1=1.2, b=0.75):
    terms = tokens((passage.get("title", "") + " " + passage.get("text", "")))
    tf = Counter(terms)
    dl = len(terms)
    q = set(query_terms)
    return sum(
        idf.get(term, 0.0)
        * (freq * (k1 + 1.0))
        / (freq + k1 * (1.0 - b + b * dl / max(avgdl, 1e-12)))
        for term, freq in tf.items()
        if term in q
    )


def prepare_representatives(train, centroids, train_labels, prior):
    """Create message-to-query representatives without using test text."""
    reps = []
    for w in range(K):
        candidates = np.where(train_labels == w)[0]
        index = int(candidates[np.argmin(((
            np.vstack([fit_query_embedding(x) for x in train[candidates]]) - centroids[w]
        ) ** 2).sum(1))])
        reps.append(train[index]["query"])

    global_w = int(np.argmax(prior))
    group_w = {}
    for group in range(K // 2):
        members = [w for w in range(K) if w // 2 == group]
        group_w[group] = max(members, key=lambda w: prior[w])

    message_queries = {0: reps[global_w]}
    for group in range(K // 2):
        message_queries[1 + group] = reps[group_w[group]]
    for w in range(K):
        message_queries[5 + w] = reps[w]
    return reps, message_queries


def fit_query_embedding(item):
    """Query-only copy of the source embedding used by the theory script."""
    import hashlib

    dim = 96
    vector = np.zeros(dim)
    text = item["query"]
    for term in TOKEN.findall(text.lower()):
        vector[int(hashlib.md5(term.encode()).hexdigest()[:8], 16) % dim] += 1
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def representative_queries(train, centroids, train_labels, prior):
    reps = []
    for w in range(K):
        candidates = np.where(train_labels == w)[0]
        vectors = np.vstack([fit_query_embedding(train[i]) for i in candidates])
        index = int(candidates[np.argmin(((vectors - centroids[w]) ** 2).sum(1))])
        reps.append(train[index]["query"])
    global_w = int(np.argmax(prior))
    message_queries = {0: reps[global_w]}
    for group in range(K // 2):
        members = [w for w in range(K) if w // 2 == group]
        group_w = max(members, key=lambda w: prior[w])
        message_queries[1 + group] = reps[group_w]
    for w in range(K):
        message_queries[5 + w] = reps[w]
    return reps, message_queries


def rank_metrics(rank):
    return {
        # All metrics are computed from the same fixed 31-passage ranking.
        # MRR@4 is explicitly truncated: it is zero when the relevant passage
        # falls below rank four.  MRR@31 uses the whole available local pool.
        "MRR@4": 1.0 / rank if rank <= 4 else 0.0,
        "MRR@31": 1.0 / rank,
        "Recall@1": float(rank <= 1),
        "Recall@3": float(rank <= 3),
        "Recall@10": float(rank <= 10),
        "NDCG@3": 1.0 / math.log2(rank + 1.0) if rank <= 3 else 0.0,
        "NDCG@10": 1.0 / math.log2(rank + 1.0) if rank <= 10 else 0.0,
    }


def precompute_output_metrics(items, message_queries, idf, avgdl):
    """Cache retrieval outcomes for every test candidate set and output symbol."""
    cached = []
    query_terms = {s: tokens(message_queries[s]) for s in range(N_S)}
    for item in items:
        candidates = [item["positive"]] + item["negative"][:N_NEG]
        per_symbol = {}
        for output_symbol in range(N_S):
            scores = [
                bm25_score(query_terms[output_symbol], passage, idf, avgdl)
                for passage in candidates
            ]
            order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
            per_symbol[output_symbol] = rank_metrics(order.index(0) + 1)
        cached.append(per_symbol)
    return cached


def expected_retrieval(cached, labels, enc, channel_matrix):
    totals = Counter()
    for per_symbol, w in zip(cached, labels):
        source_symbol = symbol(int(w), enc[int(w)])
        for output_symbol, probability in enumerate(channel_matrix[source_symbol]):
            if probability <= 0.0:
                continue
            for key, value in per_symbol[output_symbol].items():
                totals[key] += probability * value
    return {key: value / len(cached) for key, value in totals.items()}


def mean_passage_length(items):
    lengths = []
    for item in items:
        for passage in [item["positive"]] + item["negative"][:N_NEG]:
            lengths.append(len(tokens(passage.get("title", "") + " " + passage.get("text", ""))))
    return float(np.mean(lengths))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spearman(xs, ys):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            rank = (i + j + 2) / 2.0
            for pos in range(i, j + 1):
                out[order[pos]] = rank
            i = j + 1
        return np.asarray(out)

    rx, ry = ranks(xs), ranks(ys)
    return float(np.corrcoef(rx, ry)[0, 1]) if len(xs) > 1 else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    data, scanned, malformed = load_data()
    train, validation, test = data[:8400], data[8400:10200], data[10200:]
    _, centroids, train_labels, prior = fit_intents(train)
    test_vectors = np.vstack([fit_query_embedding(item) for item in test])
    test_labels = ((test_vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(2).argmin(1)
    reps, message_queries = representative_queries(train, centroids, train_labels, prior)

    # The index statistics and all query representatives are training-only.
    idf = build_bm25_index(train)
    avgdl = mean_passage_length(train)
    ql, qe = build_channels("Bob-better")[:2]
    theory_rows = []
    all_encodings = list(product(range(3), repeat=K))
    for encoding_id, enc in enumerate(all_encodings):
        metrics = exact_metrics(enc, prior, ql, qe)
        metrics.update({"encoding_id": encoding_id, "encoding": " ".join(map(str, enc))})
        theory_rows.append(metrics)
    boundary = pareto(theory_rows)
    boundary_ids = {int(row["encoding_id"]) for row in boundary}

    cached_metrics = precompute_output_metrics(test, message_queries, idf, avgdl)

    def evaluate_set(encoding_id, enc, label):
        theory = theory_rows[encoding_id]
        retrieval = expected_retrieval(cached_metrics, test_labels, enc, ql)
        row = {
            "strategy": label,
            "encoding_id": encoding_id,
            "encoding": " ".join(map(str, enc)),
            "D_L": theory["D_B"],
            "D_E": theory["D_E"],
            "L": theory["L"],
            "I_L": theory["I_B"],
            "I_E": theory["I_E"],
            "S_e": theory["Se"],
            "queries": len(test),
            **retrieval,
        }
        return row

    baselines = []
    baselines.append((0, tuple(0 for _ in range(K)), "Coarse-only"))
    baselines.append((int(sum(3 ** i for i in range(K))), tuple(1 for _ in range(K)), "Group-only"))
    baselines.append((sum(2 * 3 ** i for i in range(K)), tuple(2 for _ in range(K)), "Fine-only"))
    for budget in (2.0, 2.5):
        feasible = [row for row in theory_rows if row["L"] <= budget + 1e-12]
        min_d = min(feasible, key=lambda row: (row["D_B"], row["L"], -row["Se"]))
        max_se = max(feasible, key=lambda row: (row["Se"], row["D_B"], row["L"]))
        baselines.append((int(min_d["encoding_id"]), tuple(map(int, min_d["encoding"].split())), f"Min-D_L at L<={budget:g}"))
        baselines.append((int(max_se["encoding_id"]), tuple(map(int, max_se["encoding"].split())), f"Max-S_e at L<={budget:g}"))

    # Oracle is not a communication method; it is an application-level upper reference.
    oracle_totals = Counter()
    for item in test:
        candidates = [item["positive"]] + item["negative"][:N_NEG]
        terms = tokens(item["query"])
        scores = [bm25_score(terms, passage, idf, avgdl) for passage in candidates]
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        for key, value in rank_metrics(order.index(0) + 1).items():
            oracle_totals[key] += value
    oracle = {key: value / len(test) for key, value in oracle_totals.items()}
    rows = [evaluate_set(encoding_id, enc, label) for encoding_id, enc, label in baselines]
    oracle_row = {
        "strategy": "Original-query oracle (reference only)", "encoding_id": "NA", "encoding": "NA",
        "D_L": "NA", "D_E": "NA", "L": "NA", "I_L": "NA", "I_E": "NA", "S_e": "NA",
        "queries": len(test), **oracle,
    }
    rows.insert(0, oracle_row)
    write_csv(OUT / "MSMARCO_Web_Search_策略对比.csv", rows)

    boundary_rows = []
    for row in boundary:
        encoding_id = int(row["encoding_id"])
        boundary_rows.append(evaluate_set(encoding_id, tuple(map(int, row["encoding"].split())), "Pareto-boundary"))
    write_csv(OUT / "MSMARCO_Web_Search_Pareto逐编码结果.csv", boundary_rows)

    corr_rows = []
    for metric in ("MRR@4", "MRR@31", "Recall@1", "Recall@3", "Recall@10", "NDCG@3", "NDCG@10"):
        corr_rows.append({
            "metric": metric,
            "n_pareto_encodings": len(boundary_rows),
            "Spearman_rho_D_L_vs_metric": spearman([r["D_L"] for r in boundary_rows], [r[metric] for r in boundary_rows]),
            "Spearman_rho_S_e_vs_metric": spearman([r["S_e"] for r in boundary_rows], [r[metric] for r in boundary_rows]),
        })
    write_csv(OUT / "MSMARCO_Web_Search_D_L与检索指标相关性.csv", corr_rows)

    with open(OUT / "MSMARCO_Web_Search_实验审计.txt", "w", encoding="utf-8") as handle:
        handle.write("MS MARCO application-level Web Search validation\n")
        handle.write(f"records_scanned={scanned}\nmalformed={malformed}\n")
        handle.write(f"sample={len(data)}\ntrain={len(train)} validation={len(validation)} test={len(test)}\n")
        handle.write(f"candidate_set=one positive passage plus {N_NEG} negative passages per test query; fixed {N_NEG + 1}-candidate local pool (the largest fixed pool supported by the currently available records)\n")
        handle.write("W/prototypes/source prior/representative queries use query text only; passages and relevance labels are evaluation-only\n")
        handle.write("retrieval_scope=query-local candidate ranking; these are not official full-corpus MS MARCO scores\n")
        handle.write("metrics=MRR@4, MRR@31, Recall@1, Recall@3, Recall@10, NDCG@3, NDCG@10; all are computed from the same 31-candidate ranking, and MRR@4 is truncated at rank four\n")
        handle.write("retriever=BM25(k1=1.2,b=0.75), one fixed IDF built from training passages\n")
        handle.write("surrogate_query=recovered semantic output symbol mapped to a representative training query\n")
        handle.write("oracle=original test query with the same fixed BM25 and candidate lists; reference only\n")
        handle.write("channel_expectation=exact sum over Q_L output symbols, no single-noise-draw estimate\n")
        handle.write(f"all_deterministic_encodings={len(all_encodings)}\npareto_encodings={len(boundary_ids)}\n")
        handle.write("The retrieval metrics validate downstream application utility; D_L remains the theory-defined MAP distortion on W.\n")
    print("OUTPUT", OUT)
    print("STRATEGIES", json.dumps(rows, ensure_ascii=False))
    print("CORRELATIONS", json.dumps(corr_rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
