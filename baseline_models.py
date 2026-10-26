"""Common query-only evaluator for the isolated 2025--2026 baselines.

The evaluator intentionally works at the interface exposed by the current
MS MARCO experiment. It does not change the existing finite-code scripts.
Each method supplies one variable-length packet of symbols from the existing
13-symbol alphabet. Channel outputs are enumerated exactly for packets of
length one or two, and packet length is part of the receiver observation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "src" / "web_search"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secrecy_region import (  # noqa: E402
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
from web_search import (  # noqa: E402
    N_NEG,
    bm25_score,
    build_bm25_index,
    fit_query_embedding,
    mean_passage_length,
    rank_metrics,
    representative_queries,
    tokens,
)


TOKEN = re.compile(r"[a-z0-9]+")
BITS_PER_SYMBOL = int(math.ceil(math.log2(N_S)))
TARGET_BUDGETS = (1.0, 1.25, 1.5, 1.75, 2.0)
REFINE_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
_REP_CACHE = {}
_OUT_CACHE = {}


@dataclass
class Context:
    train: list[dict]
    validation: list[dict]
    test: list[dict]
    train_labels: np.ndarray
    validation_labels: np.ndarray
    test_labels: np.ndarray
    train_prior: np.ndarray
    centroids: np.ndarray
    reps: list[str]
    idf: dict[str, float]
    avgdl: float
    channel_l: np.ndarray
    channel_e: np.ndarray
    scanned: int
    malformed: int


def build_context() -> Context:
    data, scanned, malformed = load_data()
    train, validation, test = data[:8400], data[8400:10200], data[10200:]
    _, centroids, train_labels, train_prior = fit_intents(train)

    def labels_for(items):
        vectors = np.vstack([fit_query_embedding(item) for item in items])
        return ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(2).argmin(1)

    validation_labels = labels_for(validation)
    test_labels = labels_for(test)
    reps, _ = representative_queries(train, centroids, train_labels, train_prior)
    idf = build_bm25_index(train)
    avgdl = mean_passage_length(train)
    channel_l, channel_e = build_channels("Bob-better")[:2]
    return Context(
        train=train,
        validation=validation,
        test=test,
        train_labels=train_labels,
        validation_labels=validation_labels,
        test_labels=test_labels,
        train_prior=train_prior,
        centroids=centroids,
        reps=reps,
        idf=idf,
        avgdl=avgdl,
        channel_l=channel_l,
        channel_e=channel_e,
        scanned=scanned,
        malformed=malformed,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def group_symbol(w: int) -> int:
    return 1 + int(w) // 2


def fine_symbol(w: int) -> int:
    return 5 + int(w)


def global_symbol() -> int:
    return 0


def query_uncertainty(items: Iterable[dict], centroids: np.ndarray) -> np.ndarray:
    vectors = np.vstack([fit_query_embedding(item) for item in items])
    distances = ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(2)
    ordered = np.sort(distances, axis=1)
    # Large distance and small nearest-centroid margin indicate that refinement
    # is useful. This uses query text and training prototypes only.
    margin = ordered[:, 1] - ordered[:, 0]
    return distances.min(1) + 0.25 * (1.0 / (margin + 1e-6))


def information_density(items: Iterable[dict], idf: dict[str, float]) -> np.ndarray:
    values = []
    for item in items:
        terms = set(tokens(item["query"]))
        values.append(len(terms) + 0.10 * sum(idf.get(term, 0.0) for term in terms))
    return np.asarray(values, dtype=float)


def tfidf_svd_embeddings(train: list[dict], other: list[list[dict]]):
    """Return a query-only learned text representation for the PLM adapter.

    This fallback is deliberately named in outputs. It is used only when a
    local PLM checkpoint is unavailable, so the generated rows cannot be
    mistaken for the published checkpoint's numbers.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_text = [x["query"] for x in train]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=20000)
    x_train = vectorizer.fit_transform(train_text)
    n_components = max(2, min(64, x_train.shape[0] - 1, x_train.shape[1] - 1))
    reducer = TruncatedSVD(n_components=n_components, random_state=SEED)
    train_emb = reducer.fit_transform(x_train)
    outputs = [train_emb]
    for items in other:
        outputs.append(reducer.transform(vectorizer.transform([x["query"] for x in items])))
    return outputs, "tfidf_svd_proxy"


def plm_embeddings(train: list[dict], other: list[list[dict]]):
    """Use a supplied local PLM if configured, otherwise an explicit proxy."""
    model_path = os.environ.get("WEB_PLM_MODEL_PATH", "").strip()
    if not model_path:
        return tfidf_svd_embeddings(train, other)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        model.eval()

        def encode(items):
            result = []
            for start in range(0, len(items), 32):
                texts = [x["query"] for x in items[start : start + 32]]
                batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
                with torch.no_grad():
                    hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                result.append(pooled.cpu().numpy())
            return np.vstack(result)

        return [encode(train)] + [encode(items) for items in other], f"local_plm:{model_path}"
    except Exception as exc:  # pragma: no cover - depends on optional local assets
        embeddings, _ = tfidf_svd_embeddings(train, other)
        return embeddings, f"tfidf_svd_proxy(plm_load_failed:{type(exc).__name__})"


def normalize_embeddings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def plm_intent_predictions(
    train_emb: np.ndarray,
    other_emb: list[np.ndarray],
    train_labels: np.ndarray,
):
    train_emb = normalize_embeddings(train_emb)
    centroids = np.vstack(
        [normalize_embeddings(train_emb[train_labels == w]).mean(0) for w in range(K)]
    )
    centroids = normalize_embeddings(centroids)

    def predict(values):
        values = normalize_embeddings(values)
        distances = 1.0 - values @ centroids.T
        order = np.argsort(distances, axis=1)
        margin = distances[np.arange(len(values)), order[:, 1]] - distances[np.arange(len(values)), order[:, 0]]
        uncertainty = distances.min(1) + 0.25 / (margin + 1e-6)
        return order[:, 0].astype(int), uncertainty

    return [predict(values) for values in other_emb]


def threshold_for_fraction(scores: np.ndarray, fraction: float) -> float:
    if fraction <= 0.0:
        return float("inf")
    if fraction >= 1.0:
        return float("-inf")
    return float(np.quantile(scores, 1.0 - fraction, method="higher"))


def make_c2f_packets(labels: np.ndarray, scores: np.ndarray, threshold: float) -> list[tuple[int, ...]]:
    packets = []
    for w, score in zip(labels, scores):
        packet = [group_symbol(int(w))]
        if score >= threshold:
            packet.append(fine_symbol(int(w)))
        packets.append(tuple(packet))
    return packets


def make_vbr_packets(labels: np.ndarray, scores: np.ndarray, threshold: float) -> list[tuple[int, ...]]:
    # The two-phase controller uses a base group message and optional fine
    # refinement. Its score is information density rather than prototype error.
    return make_c2f_packets(labels, scores, threshold)


def make_plm_packets(labels: np.ndarray, scores: np.ndarray, threshold: float) -> list[tuple[int, ...]]:
    return make_c2f_packets(labels, scores, threshold)


def output_distribution(packet: tuple[int, ...], channel: np.ndarray):
    key = (tuple(packet), id(channel))
    if key in _OUT_CACHE:
        return _OUT_CACHE[key]
    states = [((), 1.0)]
    for source in packet:
        updated = []
        for prefix, probability in states:
            for output, p in enumerate(channel[source]):
                if p > 0.0:
                    updated.append((prefix + (int(output),), probability * float(p)))
        states = updated
    _OUT_CACHE[key] = states
    return states


def joint_observations(labels: np.ndarray, packets: list[tuple[int, ...]], channel: np.ndarray):
    joint = defaultdict(float)
    for w, packet in zip(labels, packets):
        for observation, probability in output_distribution(packet, channel):
            joint[(int(w), observation)] += probability / len(labels)
    return joint


def mutual_information(joint: dict[tuple[int, tuple[int, ...]], float]) -> float:
    p_w = defaultdict(float)
    p_y = defaultdict(float)
    for (w, observation), probability in joint.items():
        p_w[w] += probability
        p_y[observation] += probability
    value = 0.0
    for (w, observation), probability in joint.items():
        if probability > 0.0:
            value += probability * math.log2(probability / (p_w[w] * p_y[observation]))
    return float(value)


def map_decoder(joint: dict[tuple[int, tuple[int, ...]], float]):
    by_observation = defaultdict(lambda: defaultdict(float))
    for (w, observation), probability in joint.items():
        by_observation[observation][w] += probability
    return {observation: max(values, key=lambda w: (values[w], -w)) for observation, values in by_observation.items()}


def representative_metrics(items: list[dict], reps: list[str], idf: dict[str, float], avgdl: float):
    cached = []
    rep_terms = [tokens(query) for query in reps]
    for item in items:
        candidates = [item["positive"]] + item["negative"][:N_NEG]
        per_w = {}
        for w, terms in enumerate(rep_terms):
            scores = [bm25_score(terms, passage, idf, avgdl) for passage in candidates]
            order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
            per_w[w] = rank_metrics(order.index(0) + 1)
        cached.append(per_w)
    return cached


def evaluate_packets(
    train_labels: np.ndarray,
    train_packets: list[tuple[int, ...]],
    eval_items: list[dict],
    eval_labels: np.ndarray,
    eval_packets: list[tuple[int, ...]],
    reps: list[str],
    idf: dict[str, float],
    avgdl: float,
    channel_l: np.ndarray,
    channel_e: np.ndarray,
    rep_cache=None,
):
    train_joint_l = joint_observations(train_labels, train_packets, channel_l)
    train_joint_e = joint_observations(train_labels, train_packets, channel_e)
    decoder_l = map_decoder(train_joint_l)
    decoder_e = map_decoder(train_joint_e)
    eval_joint_l = joint_observations(eval_labels, eval_packets, channel_l)
    eval_joint_e = joint_observations(eval_labels, eval_packets, channel_e)

    def accuracy_and_rows(joint, decoder):
        correct = sum(probability for (w, observation), probability in joint.items() if decoder.get(observation, 0) == w)
        return float(correct), float(1.0 - correct)

    correct_l, d_l = accuracy_and_rows(eval_joint_l, decoder_l)
    correct_e, d_e = accuracy_and_rows(eval_joint_e, decoder_e)
    if rep_cache is None:
        rep_cache = representative_metrics(eval_items, reps, idf, avgdl)
    totals = Counter()
    per_query = []
    for index, (item, w, packet) in enumerate(zip(eval_items, eval_labels, eval_packets)):
        row_totals = Counter()
        row_correct_l = 0.0
        row_correct_e = 0.0
        for observation, probability in output_distribution(packet, channel_l):
            decoded = decoder_l.get(observation, 0)
            row_correct_l += probability * float(decoded == int(w))
            for key, value in rep_cache[index][decoded].items():
                totals[key] += probability * value
                row_totals[key] += probability * value
        for observation, probability in output_distribution(packet, channel_e):
            row_correct_e += probability * float(decoder_e.get(observation, 0) == int(w))
        per_query.append(
            {
                "query_index": index,
                "true_w": int(w),
                "packet_length": len(packet),
                "packet_symbols": " ".join(map(str, packet)),
                "Bob_W_accuracy_expected": row_correct_l,
                "Eve_W_accuracy_expected": row_correct_e,
                "D_L_expected": 1.0 - row_correct_l,
                "D_E_expected": 1.0 - row_correct_e,
                "Bob_MRR@31": row_totals["MRR@31"],
                "Bob_Recall@1": row_totals["Recall@1"],
                "Bob_Recall@10": row_totals["Recall@10"],
                "Bob_NDCG@10": row_totals["NDCG@10"],
            }
        )
    retrieval = {key: float(value / len(eval_items)) for key, value in totals.items()}
    i_l = mutual_information(eval_joint_l)
    i_e = mutual_information(eval_joint_e)
    packet_lengths = np.asarray([len(packet) for packet in eval_packets], dtype=float)
    summary = {
        "L_common": float(packet_lengths.mean()),
        "payload_bits": float(packet_lengths.mean() * BITS_PER_SYMBOL),
        "packet_length_p25": float(np.quantile(packet_lengths, 0.25)),
        "packet_length_median": float(np.quantile(packet_lengths, 0.50)),
        "packet_length_p75": float(np.quantile(packet_lengths, 0.75)),
        "packet_length_max": int(packet_lengths.max()),
        "D_L": d_l,
        "D_E": d_e,
        "Bob_W_accuracy": 1.0 - d_l,
        "Eve_W_accuracy": 1.0 - d_e,
        "Bob_Eve_accuracy_gap": d_e - d_l,
        "I_W_L": i_l,
        "I_W_E": i_e,
        "S_e": i_l - i_e,
        **retrieval,
    }
    return summary, per_query


def evaluate_policy_on_split(ctx: Context, train_packets, eval_items, eval_labels, eval_packets, rep_cache=None):
    if rep_cache is None:
        cache_key = (id(eval_items), id(ctx.reps), id(ctx.idf))
        rep_cache = _REP_CACHE.get(cache_key)
        if rep_cache is None:
            rep_cache = representative_metrics(eval_items, ctx.reps, ctx.idf, ctx.avgdl)
            _REP_CACHE[cache_key] = rep_cache
    return evaluate_packets(
        ctx.train_labels,
        train_packets,
        eval_items,
        eval_labels,
        eval_packets,
        ctx.reps,
        ctx.idf,
        ctx.avgdl,
        ctx.channel_l,
        ctx.channel_e,
        rep_cache,
    )


def paper_reference_candidates(ctx: Context):
    rows = []
    all_rows = []
    for encoding_id, encoding in enumerate(product(range(3), repeat=K)):
        metrics = exact_metrics(encoding, ctx.train_prior, ctx.channel_l, ctx.channel_e)
        metrics.update({"encoding_id": encoding_id, "encoding_tuple": encoding})
        all_rows.append(metrics)
    for budget in TARGET_BUDGETS:
        feasible = [row for row in all_rows if row["L"] <= budget + 1e-12]
        if not feasible:
            continue
        security_unaware = min(feasible, key=lambda row: (row["D_B"], row["L"], -row["Se"]))
        proposed = max(feasible, key=lambda row: (row["Se"], row["D_B"], row["L"]))
        rows.extend(
            [
                ("Security-unaware finite SC", budget, security_unaware),
                ("Proposed secrecy-aware finite SC", budget, proposed),
            ]
        )
    return rows


def paper_packets(labels: np.ndarray, encoding: tuple[int, ...]):
    return [tuple([symbol(int(w), int(encoding[int(w)]))]) for w in labels]


def paper_eval(ctx: Context, labels_train, labels_eval, items_eval, encoding):
    train_packets = paper_packets(ctx.train_labels, encoding)
    eval_packets = paper_packets(labels_eval, encoding)
    summary, per_query = evaluate_policy_on_split(ctx, train_packets, items_eval, labels_eval, eval_packets)
    mode_costs = np.asarray([COST[int(encoding[int(w)])] for w in labels_eval], dtype=float)
    summary["L_common"] = float(mode_costs.mean())
    summary["payload_bits"] = float(mode_costs.mean() * BITS_PER_SYMBOL)
    summary["paper_cost_L"] = float(mode_costs.mean())
    return summary, per_query
