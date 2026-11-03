import re
from typing import Dict, List, Set, Tuple
import pandas as pd
from nltk.stem import PorterStemmer

# Standard English stop words
STOP_WORDS: Set[str] = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 
    'aren', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 
    'but', 'by', 'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 
    'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 
    'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 
    'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 
    'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 
    's', 'same', 'she', 'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 
    'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 
    'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 
    'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 'yourself'
}

class InvertedIndex:
    """
    Inverted Index data structure with Porter Stemming and Stop-word Removal
    for efficient sparse lexical information retrieval (Unit 2).
    
    Structure:
        postings: Dict[term -> Dict[movie_id, term_frequency]]
        doc_lengths: Dict[movie_id -> total_stemmed_terms]
        doc_metadata: Dict[movie_id -> metadata_dict]
    """
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.stemmer = PorterStemmer()
        self.postings: Dict[str, Dict[int, int]] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.doc_metadata: Dict[int, Dict] = {}
        self.total_docs: int = len(self.movies_df)
        
        self._build_index()

    def tokenize_and_stem(self, text: str) -> List[str]:
        """
        Tokenizes raw text, removes stop words, and applies Porter Stemmer.
        Example: 'steals corporate secrets' -> ['steal', 'corpor', 'secret']
        """
        if not isinstance(text, str):
            return []
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        stemmed_tokens = [
            self.stemmer.stem(t) for t in tokens if t not in STOP_WORDS
        ]
        return stemmed_tokens

    def _build_index(self):
        """Builds postings lists mapping stemmed terms to movie IDs and frequencies."""
        for _, row in self.movies_df.iterrows():
            m_id = int(row['movie_id'])
            
            # Composite document
            composite_text = f"{row['title']} {row['genres']} {row['director']} {row['cast']} {row['plot_synopsis']} {row['tags']}"
            tokens = self.tokenize_and_stem(composite_text)
            
            self.doc_lengths[m_id] = len(tokens)
            self.doc_metadata[m_id] = {
                "title": row['title'],
                "release_year": int(row['release_year']),
                "genres": row['genres'],
                "director": row['director'],
                "vote_average": float(row['vote_average']),
                "poster_url": row.get('poster_url', '')
            }
            
            for token in tokens:
                if token not in self.postings:
                    self.postings[token] = {}
                self.postings[token][m_id] = self.postings[token].get(m_id, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths.values()) / max(1, self.total_docs)

    def get_postings(self, term: str) -> Dict[int, int]:
        """Returns the postings dictionary {movie_id: term_freq} for a given raw or stemmed term."""
        stemmed_term = self.stemmer.stem(term.lower())
        return self.postings.get(stemmed_term, {})

    def boolean_and_query(self, query_terms: List[str]) -> List[int]:
        """
        Boolean AND retrieval: finds movies containing ALL query terms.
        Intersects postings lists of all terms.
        """
        if not query_terms:
            return []
        
        stemmed = [self.stemmer.stem(t.lower()) for t in query_terms if t.lower() not in STOP_WORDS]
        if not stemmed:
            return []
        
        # Start with the postings of the rarest term (optimization)
        stemmed_sorted = sorted(stemmed, key=lambda t: len(self.postings.get(t, {})))
        result_docs = set(self.postings.get(stemmed_sorted[0], {}).keys())
        
        for term in stemmed_sorted[1:]:
            postings_keys = set(self.postings.get(term, {}).keys())
            result_docs = result_docs.intersection(postings_keys)
            if not result_docs:
                break
                
        return sorted(list(result_docs))

    def boolean_or_query(self, query_terms: List[str]) -> List[int]:
        """
        Boolean OR retrieval: finds movies containing ANY of the query terms.
        Unions postings lists.
        """
        stemmed = [self.stemmer.stem(t.lower()) for t in query_terms if t.lower() not in STOP_WORDS]
        result_docs = set()
        for term in stemmed:
            result_docs.update(self.postings.get(term, {}).keys())
        return sorted(list(result_docs))

    def get_term_statistics(self, term: str) -> Dict:
        """Returns term frequency (TF across catalog) and document frequency (DF)."""
        stemmed = self.stemmer.stem(term.lower())
        postings = self.postings.get(stemmed, {})
        return {
            "raw_term": term,
            "stemmed_term": stemmed,
            "document_frequency": len(postings),
            "collection_frequency": sum(postings.values()),
            "matching_movies": [self.doc_metadata[m_id]['title'] for m_id in postings.keys()]
        }

