import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    """
    Content-Based Filtering using TF-IDF Vector Space Model and Cosine Similarity.
    Builds item document representations from genres, director, cast, synopsis, and tags.
    """
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.movie_id_to_idx = {row['movie_id']: idx for idx, row in self.movies_df.iterrows()}
        self.idx_to_movie_id = {idx: row['movie_id'] for idx, row in self.movies_df.iterrows()}
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self._build_item_corpus_and_matrix()

    def _build_item_corpus_and_matrix(self):
        """Constructs a weighted composite document for each movie and computes TF-IDF."""
        corpus = []
        for _, row in self.movies_df.iterrows():
            # Genre & Director receive higher weighting by duplication
            genres_weighted = f" {row['genres']} " * 3
            director_weighted = f" {row['director']} " * 2
            cast_part = f" {row['cast']} "
            plot_part = f" {row['plot_synopsis']} "
            tags_part = f" {row['tags']} " * 2
            
            doc = f"{row['title']} {genres_weighted} {director_weighted} {cast_part} {plot_part} {tags_part}"
            corpus.append(doc)
            
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        # Precompute item-item similarity matrix for fast retrieval and MMR diversity
        self.item_item_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_similar_items(self, movie_id: int, top_k: int = 5):
        """Finds top-K most similar movies to a given movie (Item-Item Content Similarity)."""
        if movie_id not in self.movie_id_to_idx:
            return []
        
        idx = self.movie_id_to_idx[movie_id]
        sim_scores = list(enumerate(self.item_item_sim[idx]))
        # Sort descending by similarity, exclude the item itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_k]
        
        results = []
        for i, score in sim_scores:
            m_id = self.idx_to_movie_id[i]
            row = self.movies_df.iloc[i]
            results.append({
                "movie_id": m_id,
                "title": row['title'],
                "score": float(score),
                "genres": row['genres'],
                "director": row['director']
            })
        return results

    def build_user_profile_vector(self, user_ratings: list):
        """
        Creates a weighted user profile vector in the TF-IDF space:
        u = sum(w_i * v_i) / sum(w_i)
        where w_i is derived from rating and watch completion ratio.
        """
        if not user_ratings:
            return None

        user_vector = np.zeros(self.tfidf_matrix.shape[1])
        total_weight = 0.0

        for r in user_ratings:
            m_id = r.get('movie_id')
            if m_id in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[m_id]
                rating = float(r.get('rating', 3.0))
                watch_ratio = float(r.get('watch_ratio', 1.0))
                
                # Center rating around baseline (3.0): high ratings push positive, low ratings discount
                weight = max(0.1, (rating / 5.0) * watch_ratio)
                item_vec = self.tfidf_matrix[idx].toarray()[0]
                user_vector += weight * item_vec
                total_weight += weight

        if total_weight > 0:
            user_vector /= total_weight
            norm = np.linalg.norm(user_vector)
            if norm > 0:
                user_vector /= norm
            return user_vector
        return None

    def score_all_items_for_user(self, user_ratings: list):
        """Computes content similarity scores [0, 1] for all items against user profile vector."""
        user_vector = self.build_user_profile_vector(user_ratings)
        scores = {}

        if user_vector is None:
            # Cold-start fallback: return uniform prior
            for m_id in self.movie_id_to_idx:
                scores[m_id] = 0.0
            return scores

        # Dot product with normalized item vectors equals cosine similarity
        sims = self.tfidf_matrix.dot(user_vector)
        for idx, score in enumerate(sims):
            m_id = self.idx_to_movie_id[idx]
            # Rescale / clip to [0, 1]
            scores[m_id] = float(np.clip(score, 0.0, 1.0))

        return scores

    def explain_recommendation(self, user_ratings: list, target_movie_id: int):
        """Generates an explainability rationale ('Because you watched X, which shares...')"""
        if not user_ratings or target_movie_id not in self.movie_id_to_idx:
            return "Recommended based on overall catalog popularity."

        target_idx = self.movie_id_to_idx[target_movie_id]
        target_genres = set([g.strip() for g in self.movies_df.iloc[target_idx]['genres'].split(',')])
        target_director = self.movies_df.iloc[target_idx]['director']

        best_source_title = None
        best_sim = -1.0
        shared_elements = []

        for r in user_ratings:
            m_id = r['movie_id']
            if m_id in self.movie_id_to_idx and r.get('rating', 0) >= 3.5:
                src_idx = self.movie_id_to_idx[m_id]
                sim = self.item_item_sim[src_idx, target_idx]
                if sim > best_sim:
                    best_sim = sim
                    src_row = self.movies_df.iloc[src_idx]
                    best_source_title = src_row['title']
                    src_genres = set([g.strip() for g in src_row['genres'].split(',')])
                    overlapping_genres = target_genres.intersection(src_genres)
                    
                    reasons = []
                    if target_director == src_row['director']:
                        reasons.append(f"directed by {target_director}")
                    if overlapping_genres:
                        reasons.append(", ".join(list(overlapping_genres)[:2]))
                    shared_elements = reasons

        if best_source_title:
            reason_str = " & ".join(shared_elements) if shared_elements else "similar themes"
            return f"Because you loved {best_source_title} ({reason_str})"
        return "Recommended based on your interest in " + ", ".join(list(target_genres)[:2])
