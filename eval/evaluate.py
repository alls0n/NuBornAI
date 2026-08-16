
#   python eval/evaluate.py
#   python eval/evaluate.py --k 10
#   python eval/evaluate.py --set eval/eval_set.json --out eval/results.csv

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  

import config
from rag_engine import engine
from metrics import score_query, average_scores

MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


def load_eval_set(path):
    with open(path) as f:
        data = json.load(f)
    queries = data.get("queries", [])
    if not queries:
        raise ValueError(f"No queries found in {path} — fill it in first.")
    return queries


def retrieved_sources(results):
    seen = []
    for r in results:
        if r["source"] not in seen:
            seen.append(r["source"])
    return seen


def run(eval_set_path, k, out_path):
    print("Loading index (embeddings + reranker, no LLM needed for eval)...")
    engine.initialize()
    print(f"Index ready — {engine.index.ntotal} chunks.\n")

    queries = load_eval_set(eval_set_path)
    print(f"Evaluating {len(queries)} questions at k={k}\n")

    all_rows = []
    mode_results = {}

    for mode in MODES:
        if mode == "hybrid_rerank" and engine.reranker is None:
            print(f"Skipping '{mode}' — USE_RERANKER is False in config.py, reranker not loaded.")
            continue

        per_query = []
        for q in queries:
            question = q["question"]
            relevant = q["relevant_sources"]
            results = engine.search(question, top_k=max(k * 3, config.RERANK_POOL), mode=mode)
            ranked_sources = retrieved_sources(results)
            scores = score_query(ranked_sources, relevant, k)
            per_query.append(scores)
            all_rows.append({"mode": mode, "question": question, **{k2: round(v, 4) for k2, v in scores.items()}})

        mode_results[mode] = average_scores(per_query)

    # comparison 
    header = f"{'mode':<16}{'precision@k':>13}{'recall@k':>11}{'mrr':>9}{'ndcg@k':>9}"
    print(header)
    print("-" * len(header))
    for mode, avg in mode_results.items():
        print(f"{mode:<16}{avg['precision']:>13.3f}{avg['recall']:>11.3f}{avg['mrr']:>9.3f}{avg['ndcg']:>9.3f}")

    if out_path:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mode", "question", "precision", "recall", "mrr", "ndcg"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nPer-question results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality across pipeline variants.")
    parser.add_argument("--set", default=str(Path(__file__).parent / "eval_set.json"), help="Path to eval set JSON")
    parser.add_argument("--k", type=int, default=config.EVAL_K, help="k for precision/recall/nDCG")
    parser.add_argument("--out", default=str(Path(__file__).parent / "results.csv"), help="Where to write per-question CSV (or '' to skip)")
    args = parser.parse_args()

    run(args.set, args.k, args.out or None)
