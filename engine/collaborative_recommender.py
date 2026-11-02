import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    """
    Model-Based Collaborative Filtering using Matrix Factorization (Truncated SVD)
    and Memory-Based Item-Item correlation fallback.
    """
    def __init__(self, ratings_df: pd.DataFrame, n_latent_factors: int = 4):
        self.ratings_df = ratings_df.copy()
        self.n_latent_factors = n_latent_factors
        self.user_ids = sorted(self.ratings_df['user_id'].unique())
        self.movie_ids = sorted(self.ratings_df['movie_id'].unique())
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.idx_to_user = {idx: uid for idx, uid in enumerate(self.user_ids)}
        
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.idx_to_movie = {idx: mid for idx, mid in enumerate(self.movie_ids)}
        
        self._fit_matrix_factorization()

    def _fit_matrix_factorization(self):
        """Constructs User-Item interaction matrix and fits Truncated SVD."""
        num_users = len(self.user_ids)
        num_items = len(self.movie_ids)
        
        # Build interaction matrix (scaled by watch_ratio if available)
        self.R = np.zeros((num_users, num_items))
        for _, row in self.ratings_df.iterrows():
            u_idx = self.user_to_idx.get(row['user_id'])
            m_idx = self.movie_to_idx.get(row['movie_id'])
            if u_idx is not None and m_idx is not None:
                rating = float(row['rating'])
                watch_ratio = float(row.get('watch_ratio', 1.0))
                # Effective interaction weight
                self.R[u_idx, m_idx] = rating * watch_ratio

        # Determine safe number of latent components (must be < min(num_users, num_items))
        n_components = min(self.n_latent_factors, min(num_users, num_items) - 1)
        if n_components < 1:
            n_components = 1
            
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        # Latent user matrix U
        self.user_latent = self.svd.fit_transform(self.R)
        # Latent item matrix V
        self.item_latent = self.svd.components_.T
        
        # Reconstructed matrix: R_hat = U * V^T
        self.R_predicted = np.dot(self.user_latent, self.item_latent.T)
        
        # Normalize predicted ratings to [0, 1] range for unified hybrid scoring
        min_val = self.R_predicted.min()
        max_val = self.R_predicted.max()
        if max_val > min_val:
            self.R_predicted_norm = (self.R_predicted - min_val) / (max_val - min_val)
        else:
            self.R_predicted_norm = np.ones_like(self.R_predicted) * 0.5

    def predict_score_for_known_user(self, user_id: int, movie_id: int) -> float:
        """Returns normalized affinity score [0, 1] for a user known in the training matrix."""
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return 0.5  # Neutral default for unknown items/users
        
        u_idx = self.user_to_idx[user_id]
        m_idx = self.movie_to_idx[movie_id]
        return float(self.R_predicted_norm[u_idx, m_idx])

    def score_all_items_for_user(self, user_id: int, live_user_ratings: list = None) -> dict:
        """
        Scores all catalog items using Collaborative Filtering.
        If user is known, reads directly from latent factorization.
        If user is dynamic/new, projects their live ratings into the latent space:
        u_dynamic = r_dynamic * V * S^-1
        """
        scores = {}
        all_movies = self.movie_ids

        # Scenario A: Known existing user in matrix
        if user_id in self.user_to_idx:
            u_idx = self.user_to_idx[user_id]
            for m_id in all_movies:
                scores[m_id] = float(self.R_predicted_norm[u_idx, self.movie_to_idx[m_id]])
            return scores

        # Scenario B: Dynamic / New user with ratings provided
        if live_user_ratings:
            dynamic_r = np.zeros((1, len(self.movie_ids)))
            for r in live_user_ratings:
                mid = r.get('movie_id')
                if mid in self.movie_to_idx:
                    dynamic_r[0, self.movie_to_idx[mid]] = float(r.get('rating', 3.0))
            
            # Project dynamic vector into existing latent space: u_latent = r * V
            user_latent_dyn = self.svd.transform(dynamic_r)
            pred = np.dot(user_latent_dyn, self.item_latent.T)[0]
            
            min_val = pred.min()
            max_val = pred.max()
            norm_pred = (pred - min_val) / (max_val - min_val) if max_val > min_val else pred
            
            for m_id in all_movies:
                scores[m_id] = float(norm_pred[self.movie_to_idx[m_id]])
            return scores

        # Scenario C: Pure Cold-Start (New user with 0 ratings)
        for m_id in all_movies:
            scores[m_id] = 0.5
        return scores
