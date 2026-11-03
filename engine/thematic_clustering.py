import os
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Set environment variable to ensure single-threaded execution on Windows without wmic warning
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

class ThematicClusterManager:
    """
    Unsupervised Text Clustering using K-Means on TF-IDF Vectors (Unit 5).
    Groups movies into semantic themes and automatically discovers distinctive cluster keywords.
    Powers thematic 'Netflix Rails' shelves.
    """
    def __init__(self, movies_df: pd.DataFrame, n_clusters: int = 4):
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.n_clusters = n_clusters
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=600
        )
        self._fit_clusters()

    def _fit_clusters(self):
        """Constructs movie documents and fits K-Means clustering."""
        corpus = []
        for _, r in self.movies_df.iterrows():
            doc = f"{r['title']} {r['genres']} {r['director']} {r['cast']} {r['plot_synopsis']} {r['tags']}"
            corpus.append(doc)
            
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        self.labels = self.kmeans.fit_predict(self.tfidf_matrix)
        self.movies_df['cluster_id'] = self.labels
        
        # Analyze top distinctive terms per cluster centroid
        self.cluster_summaries: Dict[int, Dict] = {}
        for c_id in range(self.n_clusters):
            centroid = self.kmeans.cluster_centers_[c_id]
            top_term_indices = centroid.argsort()[-6:][::-1]
            top_terms = [self.feature_names[i] for i in top_term_indices]
            
            movies_in_cluster = self.movies_df[self.movies_df['cluster_id'] == c_id]
            
            # Generate human-readable thematic label
            label = self._generate_cluster_label(top_terms, movies_in_cluster)
            
            self.cluster_summaries[c_id] = {
                "cluster_id": c_id,
                "label": label,
                "top_keywords": top_terms,
                "movie_count": len(movies_in_cluster),
                "movies": movies_in_cluster[['movie_id', 'title', 'genres', 'director', 'vote_average', 'poster_url']].to_dict('records')
            }

    def _generate_cluster_label(self, top_terms: List[str], cluster_df: pd.DataFrame) -> str:
        """Derives a descriptive thematic shelf title from centroid terms."""
        terms_joined = " ".join(top_terms).lower()
        if any(w in terms_joined for w in ['sci', 'space', 'spider', 'multiverse', 'cyberpunk']):
            return "Sci-Fi, Cyberpunk & Multiverse Epics"
        elif any(w in terms_joined for w in ['crime', 'mafia', 'godfather', 'mobsters', 'corleone']):
            return "Gritty Crime Syndicates & Mob Classics"
        elif any(w in terms_joined for w in ['family', 'animation', 'spirits', 'mexico', 'music']):
            return "Imaginative Animation, Mystery & Wonder"
        elif any(w in terms_joined for w in ['comedy', 'concierge', 'hotel', 'parasite']):
            return "Satirical Dark Comedies & High Drama"
        else:
            return f"Thematic Spotlight: {', '.join(top_terms[:3]).title()}"

    def get_thematic_shelves(self) -> List[Dict]:
        """Returns all thematic shelves formatted for homepage display."""
        return list(self.cluster_summaries.values())

    def predict_cluster_for_text(self, text: str) -> int:
        """Assigns an arbitrary text synopsis to the closest cluster centroid."""
        vec = self.vectorizer.transform([text])
        return int(self.kmeans.predict(vec)[0])

