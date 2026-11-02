import numpy as np
from typing import List, Dict

class MMRDiversifier:
    """
    Maximal Marginal Relevance (MMR) Re-Ranking.
    Balances individual item relevance with Intra-List Diversity to avoid recommendation bubbles.
    
    Formula:
        MMR = argmax_{d in C \\ S} [ lambda * Rel(d) - (1 - lambda) * max_{d_j in S} Sim(d, d_j) ]
    """
    def __init__(self, item_sim_matrix: np.ndarray, movie_id_to_idx: dict):
        self.item_sim_matrix = item_sim_matrix
        self.movie_id_to_idx = movie_id_to_idx

    def rerank(
        self,
        candidate_items: List[Dict],
        top_k: int = 10,
        lambda_param: float = 0.70
    ) -> List[Dict]:
        """
        Reranks a list of candidate dictionaries containing 'movie_id' and 'relevance_score'.
        
        Parameters:
            candidate_items: list of dicts with 'movie_id' and 'relevance_score'
            top_k: number of diverse items to select
            lambda_param: trade-off between pure relevance (1.0) and pure novelty/diversity (0.0).
                          Standard industry sweet spot is 0.65 - 0.75.
        """
        if not candidate_items:
            return []

        # Filter candidates to only those in the similarity matrix
        valid_candidates = [
            c for c in candidate_items if c['movie_id'] in self.movie_id_to_idx
        ]
        if not valid_candidates:
            return candidate_items[:top_k]

        selected = []
        unselected = valid_candidates.copy()

        # Step 1: Select the single most relevant item as seed
        unselected.sort(key=lambda x: x['relevance_score'], reverse=True)
        first_pick = unselected.pop(0)
        selected.append(first_pick)

        # Step 2: Iteratively select items maximizing the MMR criterion
        while len(selected) < top_k and unselected:
            best_mmr_score = -float('inf')
            best_item_idx = -1

            for i, cand in enumerate(unselected):
                cand_idx = self.movie_id_to_idx[cand['movie_id']]
                relevance = cand['relevance_score']

                # Compute maximum similarity to any already selected item
                max_sim_to_selected = max([
                    self.item_sim_matrix[cand_idx, self.movie_id_to_idx[s['movie_id']]]
                    for s in selected
                ])

                # MMR trade-off formula
                mmr_score = (lambda_param * relevance) - ((1.0 - lambda_param) * max_sim_to_selected)

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_item_idx = i

            if best_item_idx != -1:
                chosen = unselected.pop(best_item_idx)
                chosen['mmr_score'] = round(best_mmr_score, 4)
                selected.append(chosen)
            else:
                break

        return selected
