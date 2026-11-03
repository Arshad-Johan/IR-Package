# 🎬 CineMatch: Information Retrieval & Recommender Engine

A complete, academic- and production-grade Information Retrieval (IR) package for movie discovery and search inspired by modern multi-stage recommendation architectures (e.g., Netflix). Aligned directly with the university **Information Retrieval (IR)** curriculum.

---

## 🌟 Key Features & IR Capabilities

1. **Inverted Index & Morphological Stemming (Unit 2):**
   * High-efficiency sparse inverted index mapping terms $\to$ postings lists.
   * Stop-word removal and **Porter Stemmer** normalization.
   * Boolean retrieval operations (`AND`, `OR`).
2. **Dual-Retrieval Engine (Unit 3):**
   * **Dense Vector Space Model (VSM):** TF-IDF term weighting and Cosine Similarity ranking over multi-field composite documents.
   * **Okapi BM25 Probabilistic Retrieval:** Term frequency saturation ($k_1$) and document length normalization ($b$) with Robertson-Spärck Jones smoothed IDF.
3. **Query Expansion (Unit 4):**
   * Domain thesaurus expansion to enrich short/sparse queries before BM25 scoring.
4. **Unsupervised Document Clustering (Unit 5):**
   * $k$-Means clustering on the TF-IDF feature space.
   * Automated centroid keyword extraction to dynamically discover and populate themed browsing shelves.
5. **Collaborative Latent Factor Modeling (Unit 5 & Unit 7):**
   * Truncated SVD matrix factorization with dynamic out-of-sample projection (folding-in) for new session users.
6. **Multi-Criteria Two-Stage Hybrid Fusion (Unit 7):**
   * Fuses content similarity ($\alpha$), collaborative affinity ($\beta$), static quality prior ($\gamma$), and dynamic environmental context modifiers (device and temporal signals).
   * **Dynamic Cold-Start Handling:** Automatically switches to content-dominant retrieval for users with $\le 1$ interaction.
7. **Search Result Diversification via MMR (Unit 3 & Unit 7):**
   * Carbonell & Goldstein's **Maximal Marginal Relevance (MMR)** algorithm to break filter bubbles and maximize intra-list diversity.
8. **Comprehensive Offline IR Benchmark Suite (Unit 4):**
   * Evaluates $\text{Precision}@K$, $\text{Recall}@K$, $F_1\text{-Measure}@K$, $\text{NDCG}@K$, $\text{Intra-List Diversity (ILD)}$, and $\text{Catalog Coverage}$.

---

## 📂 Repository File Structure

```text
d:/Semester 9/IR Package/
│
├── data_loader.py                     # Catalog builder & interaction matrix generator
├── movies.csv                         # 20 curated movie documents with rich metadata & posters
├── ratings.csv                        # User-item interaction matrix with implicit watch ratios
│
├── engine/                            # Core Information Retrieval Engine Modules
│   ├── __init__.py                    # Package exports
│   ├── inverted_index.py              # Inverted Index, Porter Stemmer & Boolean Query Ops (Unit 2)
│   ├── content_recommender.py         # TF-IDF Vector Space Model & Cosine Similarity (Unit 3)
│   ├── bm25_retriever.py              # Okapi BM25 Probabilistic Retrieval & Query Expansion (Unit 3 & 4)
│   ├── thematic_clustering.py         # K-Means Document Clustering & Shelf Discovery (Unit 5)
│   ├── collaborative_recommender.py   # Truncated SVD Matrix Factorization & Folding-In (Unit 5 & 7)
│   ├── context_filter.py              # Device & Temporal contextual boost modifiers (Unit 7)
│   ├── mmr_diversity.py               # Maximal Marginal Relevance (MMR) re-ranker
│   └── hybrid_engine.py               # Two-Stage Pipeline & Thematic Netflix Rails Generator
│
├── evaluate.py                        # Offline IR evaluation suite (Precision, Recall, F1, NDCG, ILD)
├── app.py                             # Interactive Streamlit Web App with live BM25 search & clusters
├── index.html                         # Standalone client-side streaming app (zero-dependency)
├── slides.html                        # 18-Slide interactive presentation deck with speaker scripts
├── presentation_deck.md               # 18-Slide presentation script & speaker allocations
└── IR_METHODS_AND_OUTPUTS.md          # Comprehensive reference: formulas, code snippets & verified outputs
```

---

## 📋 Course Syllabus Alignment Matrix

| Syllabus Unit | Course Topics | CineMatch Implementation | Source File |
| :--- | :--- | :--- | :--- |
| **Unit 2: Text Representation** | Porter stemmer, stop-words, inverted indices, sparse vectors | Lexical tokenization, stop-word elimination, Porter stemming, and postings lists | [`engine/inverted_index.py`](file:///d:/Semester%209/IR%20Package/engine/inverted_index.py) |
| **Unit 3: Retrieval Models** | VSM, TF-IDF weighting, cosine similarity | Dense term-document matrix, Rocchio user query centroid, cosine ranking | [`engine/content_recommender.py`](file:///d:/Semester%209/IR%20Package/engine/content_recommender.py) |
| **Unit 3: Retrieval Models** | Probabilistic Models (Okapi BM25) | BM25 probabilistic retrieval with length normalization & saturation | [`engine/bm25_retriever.py`](file:///d:/Semester%209/IR%20Package/engine/bm25_retriever.py) |
| **Unit 4: Query Processing** | Query expansion | Domain thesaurus expansion for sparse keyword queries | [`engine/bm25_retriever.py`](file:///d:/Semester%209/IR%20Package/engine/bm25_retriever.py) |
| **Unit 4: IR Evaluation** | Precision, Recall, $F$-measure, NDCG | Quantitative benchmark suite with harmonic mean $F_1$-measure | [`evaluate.py`](file:///d:/Semester%209/IR%20Package/evaluate.py) |
| **Unit 5: Text Clustering** | $k$-Means clustering, centroid keywords | Unsupervised semantic clustering to generate dynamic themed shelves | [`engine/thematic_clustering.py`](file:///d:/Semester%209/IR%20Package/engine/thematic_clustering.py) |
| **Unit 5: Dimension Reduction** | LSI / SVD matrix factorization | Low-rank latent factor decomposition of interaction matrices | [`engine/collaborative_recommender.py`](file:///d:/Semester%209/IR%20Package/engine/collaborative_recommender.py) |
| **Unit 7: Information Filtering** | Content-based, Collaborative, Relevance feedback | Two-stage hybrid engine with implicit watch-ratio relevance feedback | [`engine/hybrid_engine.py`](file:///d:/Semester%209/IR%20Package/engine/hybrid_engine.py) |

---

## 🚀 Quick Start Guide

### Option 1: Interactive Streamlit Web Application
Run the web dashboard featuring live sliders for $\alpha$, $\beta$, $\gamma$, diversity $\lambda$, real-time rating updates, and an interactive **Okapi BM25 + $k$-Means cluster** exploration tab:
```bash
streamlit run app.py
```

### Option 2: Offline IR Benchmark Evaluation
Run the automated benchmark evaluation:
```bash
python evaluate.py
```
Outputs the performance table across all configurations ($K = 5$):
```text
========================================================================================
--- INFORMATION RETRIEVAL BENCHMARK EVALUATION (K = 5) ---
========================================================================================
                  Paradigm  Precision@5  Recall@5   F1@5  NDCG@5  Intra-List Diversity Catalog Coverage
     1. Content-Based Only       0.8286    1.0000 0.8730  1.0000                0.9371           100.0%
     2. Collaborative Only       0.7714    0.9429 0.8159  0.9976                0.9379            90.0%
      3. Hybrid (Standard)       0.8286    1.0000 0.8730  1.0000                0.9373           100.0%
4. Hybrid + MMR (Proposed)       0.8286    1.0000 0.8730  1.0000                0.9396           100.0%
========================================================================================
```

### Option 3: Double-Click Standalone Web App
Simply open [index.html](file:///d:/Semester%209/IR%20Package/index.html) in any modern web browser. It runs client-side with zero Python dependencies.

### Option 4: Presentation Deck & Speaker Scripts
Open [slides.html](file:///d:/Semester%209/IR%20Package/slides.html) in your browser. Use Arrow Keys to navigate, and press `S` to toggle speaker notes. Complete scripts are also documented in [`presentation_deck.md`](file:///d:/Semester%209/IR%20Package/presentation_deck.md).

---

## 📐 Mathematical Formulations

### 1. Vector Space Model & Content Cosine Similarity
$$\text{Sim}(\mathbf{u}, \mathbf{v}_j) = \frac{\mathbf{u} \cdot \mathbf{v}_j}{\|\mathbf{u}\|_2 \|\mathbf{v}_j\|_2}$$
where user profile $\mathbf{u}$ is formulated as a Rocchio centroid query:
$$\mathbf{u} = \frac{\sum_{i \in H_u} w_i \mathbf{v}_i}{\sum_{i \in H_u} w_i}, \quad w_i = \max\left(0.1, \frac{r_i}{5.0} \cdot c_i\right)$$

### 2. Okapi BM25 Probabilistic Ranking
$$\text{BM25}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
$$\text{IDF}(q) = \ln\left(\frac{N - n(q) + 0.5}{n(q) + 0.5} + 1.0\right)$$

### 3. Latent Factor Matrix Factorization (Truncated SVD)
$$R \approx U \Sigma V^T$$
Dynamic out-of-sample projection for session users:
$$\mathbf{u}_{\text{latent}} = \mathbf{r}_{\text{dynamic}} \cdot V, \quad \hat{\mathbf{r}} = \mathbf{u}_{\text{latent}} \cdot V^T$$

### 4. $k$-Means Document Clustering
$$\min \sum_{j=1}^k \sum_{d_i \in C_j} \| \mathbf{v}_i - \boldsymbol{\mu}_j \|^2$$

### 5. Multi-Criteria Hybrid Fusion
$$\text{Relevance}(u, i) = \alpha \cdot S_{\text{Content}}(u, i) + \beta \cdot S_{\text{Collab}}(u, i) + \gamma \cdot \text{QualityPrior}(i) + \Delta_{\text{Context}}$$

### 6. Search Result Diversification (Maximal Marginal Relevance)
$$\text{MMR} = \arg\max_{d \in \mathcal{C} \setminus \mathcal{S}} \left[ \lambda \cdot \text{Rel}(d) - (1 - \lambda) \max_{d_j \in \mathcal{S}} \text{Sim}(d, d_j) \right]$$

### 7. Evaluation Metrics ($F_1$-Measure & NDCG)
$$F_1@K = \frac{2 \cdot \text{Precision}@K \cdot \text{Recall}@K}{\text{Precision}@K + \text{Recall}@K}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

---

## 📖 In-Depth Technical Reference
For code snippets and terminal outputs for every method, see [`IR_METHODS_AND_OUTPUTS.md`](file:///d:/Semester%209/IR%20Package/IR_METHODS_AND_OUTPUTS.md).
