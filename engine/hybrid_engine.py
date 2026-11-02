import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from engine.content_recommender import ContentBasedRecommender
from engine.collaborative_recommender import CollaborativeRecommender
from engine.context_filter import ContextFilter
from engine.mmr_diversity import MMRDiversifier

class HybridRecommenderEngine:
    """
    Two-Stage Information Retrieval Hybrid Recommendation Pipeline.
    
    Stage 1: Candidate Generation (Retrieval Stage: High Recall)
    Stage 2: Scoring & Hybrid Fusion (Content + Collaborative + Quality + Context)
    Stage 3: Re-ranking & Diversity (MMR) & Themed Netflix Rails Generation
    """
    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.ratings_df = ratings_df.copy()
        
        # Sub-engines
        self.content_engine = ContentBasedRecommender(self.movies_df)
        self.collab_engine = CollaborativeRecommender(self.ratings_df)
        self.context_filter = ContextFilter(self.movies_df)
        self.diversifier = MMRDiversifier(
            item_sim_matrix=self.content_engine.item_item_sim,
            movie_id_to_idx=self.content_engine.movie_id_to_idx
        )
        
        # Normalized catalog quality prior (vote_average / 10.0)
        self.movies_df['quality_prior'] = self.movies_df['vote_average'] / 10.0

    def recommend(
        self,
        user_id: int,
        live_ratings: Optional[List[Dict]] = None,
        alpha: float = 0.45,       # Content-based weight
        beta: float = 0.40,        # Collaborative weight
        gamma: float = 0.15,       # Overall quality / rating prior weight
        device: str = "smart_tv",
        time_of_day: str = "evening",
        day_of_week: str = "friday",
        genre_filter: Optional[str] = None,
        top_k: int = 10,
        enable_mmr: bool = True,
        mmr_lambda: float = 0.70,
        filter_watched: bool = True
    ) -> List[Dict]:
        """
        Executes the full two-stage hybrid recommendation pipeline.
        """
        # Determine effective user ratings (from database or dynamic session)
        if live_ratings is not None:
            user_ratings = live_ratings
        else:
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id].to_dict('records')

        watched_movie_ids = set([r['movie_id'] for r in user_ratings])
        num_interactions = len(user_ratings)

        # Dynamic Switching Hybrid Rule for Cold-Start:
        # If user has <= 1 rating, Collaborative Filtering lacks sufficient signal.
        if num_interactions <= 1:
            eff_alpha = 0.80
            eff_beta = 0.00
            eff_gamma = 0.20
            cold_start_active = True
        else:
            # Normalize user-specified weights so they sum to 1.0
            total_w = alpha + beta + gamma
            if total_w <= 0:
                total_w = 1.0
            eff_alpha = alpha / total_w
            eff_beta = beta / total_w
            eff_gamma = gamma / total_w
            cold_start_active = False

        # --- STAGE 1: Candidate Generation & Scoring ---
        content_scores = self.content_engine.score_all_items_for_user(user_ratings)
        collab_scores = self.collab_engine.score_all_items_for_user(user_id, user_ratings)

        candidates = []
        for _, row in self.movies_df.iterrows():
            m_id = row['movie_id']

            # Option to filter already watched movies
            if filter_watched and m_id in watched_movie_ids:
                continue

            c_score = content_scores.get(m_id, 0.0)
            cf_score = collab_scores.get(m_id, 0.5)
            q_score = row['quality_prior']

            # --- STAGE 2: Multi-Criteria Hybrid Fusion ---
            base_relevance = (eff_alpha * c_score) + (eff_beta * cf_score) + (eff_gamma * q_score)

            # Apply Context Boost
            context_boost = self.context_filter.compute_context_modifier(
                movie_id=m_id,
                device=device,
                time_of_day=time_of_day,
                day_of_week=day_of_week,
                genre_filter=genre_filter
            )

            final_relevance = float(np.clip(base_relevance + context_boost, 0.0, 1.0))

            explanation = self.content_engine.explain_recommendation(user_ratings, m_id)

            candidates.append({
                "movie_id": m_id,
                "title": row['title'],
                "release_year": int(row['release_year']),
                "genres": row['genres'],
                "director": row['director'],
                "cast": row['cast'],
                "plot_synopsis": row['plot_synopsis'],
                "vote_average": float(row['vote_average']),
                "runtime_min": int(row['runtime_min']),
                "poster_url": row['poster_url'],
                "content_score": round(c_score, 3),
                "collab_score": round(cf_score, 3),
                "quality_prior": round(q_score, 3),
                "context_boost": round(context_boost, 3),
                "relevance_score": round(final_relevance, 4),
                "explanation": explanation,
                "cold_start_fallback": cold_start_active
            })

        # Sort candidates descending by score
        candidates.sort(key=lambda x: x['relevance_score'], reverse=True)

        # --- STAGE 3: MMR Diversity Re-ranking ---
        if enable_mmr:
            final_recs = self.diversifier.rerank(
                candidate_items=candidates,
                top_k=top_k,
                lambda_param=mmr_lambda
            )
        else:
            final_recs = candidates[:top_k]

        return final_recs

    def generate_netflix_rails(
        self,
        user_id: int,
        live_ratings: Optional[List[Dict]] = None,
        device: str = "smart_tv",
        time_of_day: str = "evening",
        day_of_week: str = "friday"
    ) -> Dict[str, List[Dict]]:
        """
        Generates personalized Netflix-style categorized rows (rails).
        """
        if live_ratings is not None:
            user_ratings = live_ratings
        else:
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id].to_dict('records')

        rails = {}

        # Rail 1: Top Picks For You (Two-Stage Hybrid + MMR Diversity)
        rails["Top Picks for You"] = self.recommend(
            user_id=user_id,
            live_ratings=user_ratings,
            alpha=0.45,
            beta=0.40,
            gamma=0.15,
            device=device,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            enable_mmr=True,
            mmr_lambda=0.70,
            top_k=6
        )

        # Rail 2: "Because You Watched {Favorite}" (Content-Based)
        if user_ratings:
            top_favorite = max(user_ratings, key=lambda x: x.get('rating', 0))
            fav_id = top_favorite['movie_id']
            fav_row = self.movies_df[self.movies_df['movie_id'] == fav_id]
            if not fav_row.empty:
                fav_title = fav_row.iloc[0]['title']
                similar_items = self.content_engine.get_similar_items(fav_id, top_k=6)
                similar_full = []
                for s in similar_items:
                    m_row = self.movies_df[self.movies_df['movie_id'] == s['movie_id']].iloc[0]
                    similar_full.append({
                        "movie_id": s['movie_id'],
                        "title": m_row['title'],
                        "release_year": int(m_row['release_year']),
                        "genres": m_row['genres'],
                        "director": m_row['director'],
                        "vote_average": float(m_row['vote_average']),
                        "poster_url": m_row['poster_url'],
                        "plot_synopsis": m_row['plot_synopsis'],
                        "relevance_score": round(s['score'], 3),
                        "explanation": f"Similar thematic DNA to {fav_title}"
                    })
                rails[f"Because You Watched {fav_title}"] = similar_full

        # Rail 3: Trending & Community Consensus (Collaborative Heavy)
        rails["Critically Acclaimed & Crowd Favorites"] = self.recommend(
            user_id=user_id,
            live_ratings=user_ratings,
            alpha=0.10,
            beta=0.55,
            gamma=0.35,
            device=device,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            enable_mmr=True,
            mmr_lambda=0.60,
            top_k=6
        )

        # Rail 4: Contextual Shelf (e.g. Quick Bites or Sci-Fi Epics)
        context_label = "Prime Time Movie Night" if time_of_day in ["evening", "night"] else "Quick Daytime Picks"
        rails[context_label] = self.recommend(
            user_id=user_id,
            live_ratings=user_ratings,
            alpha=0.30,
            beta=0.30,
            gamma=0.40,
            device=device,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            enable_mmr=False,
            top_k=6
        )

        return rails
