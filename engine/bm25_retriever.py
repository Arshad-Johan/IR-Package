import math
from typing import Dict, List, Optional
import pandas as pd
from engine.inverted_index import InvertedIndex

class BM25Retriever:
    """
    Okapi BM25 Probabilistic Information Retrieval Model (Unit 3).
    
    Formula:
        BM25(D, Q) = sum_{q in Q} IDF(q) * [ f(q, D) * (k1 + 1) ] / [ f(q, D) + k1 * (1 - b + b * (|D| / avgdl)) ]
        where IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1.0 )
    """
    def __init__(
        self,
        movies_df: pd.DataFrame,
        k1: float = 1.5,
        b: float = 0.75,
        inverted_index: Optional[InvertedIndex] = None
    ):
        self.k1 = k1
        self.b = b
        self.index = inverted_index if inverted_index is not None else InvertedIndex(movies_df)
        self.N = self.index.total_docs
        self.avgdl = self.index.avg_doc_len

    def compute_idf(self, term: str) -> float:
        """Computes Robertson-Spärck Jones IDF with smoothing."""
        stemmed = self.index.stemmer.stem(term.lower())
        n_q = len(self.index.postings.get(stemmed, {}))
        if n_q == 0:
            return 0.0
        # Smoothed BM25 IDF
        return math.log(((self.N - n_q + 0.5) / (n_q + 0.5)) + 1.0)

    def search(
        self,
        query: str,
        top_k: int = 5,
        query_expansion: bool = False
    ) -> List[Dict]:
        """
        Executes a BM25 ranked search for a user text query.
        Returns top-K matching documents with BM25 score, title, and term highlights.
        """
        raw_tokens = query.strip().split()
        stemmed_query = self.index.tokenize_and_stem(query)
        
        # Simple Thesaurus-based Query Expansion (Unit 4)
        expanded_terms = []
        if query_expansion:
            SYNONYMS = {
                'heist': ['steal', 'robberi', 'thief', 'crime'],
                'space': ['relativ', 'wormhol', 'orbit', 'astronaut'],
                'futur': ['cyberpunk', 'dystopia', 'ai', 'simul'],
                'crime': ['mafia', 'gangster', 'syndic', 'corleon'],
                'mind': ['subconsci', 'hallucin', 'trauma', 'dream']
            }
            for t in stemmed_query:
                if t in SYNONYMS:
                    expanded_terms.extend(SYNONYMS[t])
            stemmed_query = list(set(stemmed_query + expanded_terms))

        if not stemmed_query:
            return []

        doc_scores: Dict[int, float] = {}
        matched_terms_per_doc: Dict[int, List[str]] = {}

        for term in stemmed_query:
            idf = self.compute_idf(term)
            if idf <= 0.0:
                continue

            postings = self.index.postings.get(term, {})
            for doc_id, freq in postings.items():
                doc_len = self.index.doc_lengths.get(doc_id, self.avgdl)
                # BM25 term weighting component
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                term_score = idf * (numerator / denominator)

                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + term_score
                if doc_id not in matched_terms_per_doc:
                    matched_terms_per_doc[doc_id] = []
                matched_terms_per_doc[doc_id].append(term)

        # Sort descending by BM25 score
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in ranked_docs:
            meta = self.index.doc_metadata.get(doc_id, {})
            results.append({
                "movie_id": doc_id,
                "title": meta.get("title", f"Movie #{doc_id}"),
                "bm25_score": round(float(score), 4),
                "release_year": meta.get("release_year"),
                "genres": meta.get("genres"),
                "director": meta.get("director"),
                "vote_average": meta.get("vote_average"),
                "poster_url": meta.get("poster_url", ""),
                "matched_terms": matched_terms_per_doc.get(doc_id, [])
            })

        return results

    def score_all_items(self, query: str) -> Dict[int, float]:
        """Returns normalized [0, 1] BM25 scores for all movies against a query string."""
        raw_res = self.search(query, top_k=self.N)
        if not raw_res:
            return {m_id: 0.0 for m_id in self.index.doc_metadata}
        
        max_score = raw_res[0]['bm25_score']
        score_dict = {m['movie_id']: (m['bm25_score'] / max_score if max_score > 0 else 0.0) for m in raw_res}
        # Fill zero for non-matching movies
        for m_id in self.index.doc_metadata:
            if m_id not in score_dict:
                score_dict[m_id] = 0.0
        return score_dict

