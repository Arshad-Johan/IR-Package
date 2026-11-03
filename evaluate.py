import numpy as np
import pandas as pd
from typing import List, Dict

from engine.hybrid_engine import HybridRecommenderEngine
from engine.bm25_retriever import BM25Retriever

def dcg_at_k(r: List[float], k: int) -> float:
    """Computes Discounted Cumulative Gain at rank K."""
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r: List[float], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank K."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max

def intra_list_diversity(movie_ids: List[int], item_sim_matrix: np.ndarray, movie_id_to_idx: dict) -> float:
    """
    Computes average pairwise distance (1 - cosine similarity) between recommended items.
    Higher value indicates greater diversity and less clustering/filter bubble.
    """
    indices = [movie_id_to_idx[mid] for mid in movie_ids if mid in movie_id_to_idx]
    n = len(indices)
    if n <= 1:
        return 0.0

    total_dist = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = item_sim_matrix[indices[i], indices[j]]
            total_dist += (1.0 - sim)
            pairs += 1

    return float(total_dist / pairs) if pairs > 0 else 0.0

def run_ir_evaluation(k: int = 5):
    """
    Evaluates Content-Based vs Collaborative Filtering vs Hybrid vs Hybrid + MMR
    across Precision@K, Recall@K, F1-Measure@K, NDCG@K, Intra-List Diversity (ILD), and Catalog Coverage.
    """
    movies_df = pd.read_csv("movies.csv")
    ratings_df = pd.read_csv("ratings.csv")
    
    engine = HybridRecommenderEngine(movies_df, ratings_df)
    sim_matrix = engine.content_engine.item_item_sim
    mid_to_idx = engine.content_engine.movie_id_to_idx
    
    all_users = sorted(ratings_df['user_id'].unique())
    catalog_size = len(movies_df)

    configurations = {
        "1. Content-Based Only": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "mmr": False},
        "2. Collaborative Only": {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "mmr": False},
        "3. Hybrid (Standard)": {"alpha": 0.45, "beta": 0.40, "gamma": 0.15, "mmr": False},
        "4. Hybrid + MMR (Proposed)": {"alpha": 0.45, "beta": 0.40, "gamma": 0.15, "mmr": True, "lambda": 0.70}
    }

    results = []

    for name, config in configurations.items():
        precisions = []
        recalls = []
        f1_scores = []
        ndcgs = []
        diversities = []
        recommended_catalog_items = set()

        for uid in all_users:
            user_ratings = ratings_df[ratings_df['user_id'] == uid]
            # Ground truth: movies rated >= 3.5 are considered relevant
            relevant_items = set(user_ratings[user_ratings['rating'] >= 3.5]['movie_id'])
            
            recs = engine.recommend(
                user_id=uid,
                alpha=config['alpha'],
                beta=config['beta'],
                gamma=config['gamma'],
                enable_mmr=config['mmr'],
                mmr_lambda=config.get('lambda', 0.70),
                filter_watched=False,
                top_k=k
            )

            rec_ids = [r['movie_id'] for r in recs]
            recommended_catalog_items.update(rec_ids)

            # Relevance indicators (1 if item in relevant_items, else 0)
            hits = [1 if mid in relevant_items else 0 for mid in rec_ids]
            
            # Precision@K
            p_at_k = sum(hits) / k if k > 0 else 0.0
            precisions.append(p_at_k)

            # Recall@K
            r_at_k = sum(hits) / len(relevant_items) if len(relevant_items) > 0 else 0.0
            recalls.append(r_at_k)

            # F1-Measure@K (Unit 4: Performance metrics)
            if (p_at_k + r_at_k) > 0:
                f1_at_k = (2.0 * p_at_k * r_at_k) / (p_at_k + r_at_k)
            else:
                f1_at_k = 0.0
            f1_scores.append(f1_at_k)

            # NDCG@K
            user_ndcg = ndcg_at_k(hits, k)
            ndcgs.append(user_ndcg)

            # Intra-List Diversity
            ild = intra_list_diversity(rec_ids, sim_matrix, mid_to_idx)
            diversities.append(ild)

        coverage = len(recommended_catalog_items) / catalog_size

        results.append({
            "Paradigm": name,
            f"Precision@{k}": round(np.mean(precisions), 4),
            f"Recall@{k}": round(np.mean(recalls), 4),
            f"F1@{k}": round(np.mean(f1_scores), 4),
            f"NDCG@{k}": round(np.mean(ndcgs), 4),
            "Intra-List Diversity": round(np.mean(diversities), 4),
            "Catalog Coverage": f"{round(coverage * 100, 1)}%"
        })

    eval_df = pd.DataFrame(results)
    print("\n" + "="*88)
    print(f"--- INFORMATION RETRIEVAL BENCHMARK EVALUATION (K = {k}) ---")
    print("="*88)
    print(eval_df.to_string(index=False))
    print("="*88 + "\n")
    return eval_df

if __name__ == "__main__":
    run_ir_evaluation(k=5)
