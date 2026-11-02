# 🎬 CineMatch: Two-Stage Hybrid Information Retrieval Recommendation Engine

A complete, production-grade Information Retrieval package for movie and television discovery inspired by Netflix's multi-stage recommendation architecture.

---

## 🌟 Key Features

1. **Two-Stage IR Pipeline:**
   - **Stage 1: Candidate Generation (Retrieval Stage)** — Fast semantic retrieval using TF-IDF / Vector Space Models & Collaborative factor neighborhoods (discards 99% of catalog in $<15\text{ms}$).
   - **Stage 2: Multi-Criteria Scoring & Fusion** — Combines content vector similarity ($\alpha$), collaborative latent affinities ($\beta$), critic quality consensus ($\gamma$), and dynamic environmental context modifiers.
   - **Stage 3: Re-ranking & Diversity (MMR)** — Maximal Marginal Relevance to eliminate recommendation "filter bubbles" and guarantee intra-list novelty.
2. **Dynamic Cold-Start Handling:**
   - Automatically detects brand-new users ($\le 1$ interaction) and switches from Collaborative to pure Content-Based + Quality Prior.
3. **Context-Aware Sensing:**
   - Adjusts scores dynamically for hardware device (Smart TV vs. Mobile phone) and temporal state (Friday prime time vs. weekday lunch).
4. **Transparent Explainability:**
   - Generates natural language rationales for each recommended title (*"Because you loved Inception (directed by Christopher Nolan)"*).
5. **Rigorous IR Evaluation Suite:**
   - Evaluates Precision@K, Recall@K, NDCG@K, Intra-List Diversity (ILD), and Catalog Coverage.

---

## 📂 Repository File Structure

```text
d:/Semester 9/IR Package/
│
├── data_loader.py                 # Ingests & creates movies.csv and ratings.csv
├── movies.csv                     # Curated movie catalog with rich metadata & posters
├── ratings.csv                    # User-item interaction matrix with watch ratios
│
├── engine/                        # Core Recommender Algorithms
│   ├── __init__.py                # Package exports
│   ├── content_recommender.py     # TF-IDF Vector Space Model & Cosine Similarity
│   ├── collaborative_recommender.py # Truncated SVD Matrix Factorization
│   ├── context_filter.py          # Device & Temporal contextual boost modifiers
│   ├── mmr_diversity.py           # Maximal Marginal Relevance (MMR) re-ranker
│   └── hybrid_engine.py           # Two-Stage Pipeline & Netflix rails generator
│
├── evaluate.py                    # Offline IR benchmark evaluation (NDCG, Precision, Recall, ILD)
├── app.py                         # Interactive Streamlit Web Application
├── index.html                     # Standalone Netflix-style web app (Double-click to open)
├── slides.html                    # 18-Slide Interactive Presentation Deck with Speaker Scripts
└── presentation_deck.md           # 18-Slide detailed script & outline for Person 1, 2, 3
```

---

## 🚀 Quick Start Guide

### Option 1: Launch the Interactive Streamlit Web App
In your terminal, run:
```bash
streamlit run app.py
```
This opens the web dashboard featuring live sliders for $\alpha$, $\beta$, $\gamma$, diversity $\lambda$, live user switching, and real-time rating updates.

### Option 2: Run the Offline IR Evaluation Benchmark
Run:
```bash
python evaluate.py
```
Outputs a comparison table:
- Content-Based Only vs.
- Collaborative Filtering Only vs.
- Standard Hybrid vs.
- Hybrid + MMR (Proposed)

### Option 3: Double-Click Standalone Web App
Simply open [index.html](file:///d:/Semester%209/IR%20Package/index.html) in Google Chrome or Microsoft Edge. It runs client-side with instant responsiveness and requires no active Python process.

### Option 4: Presentation Deck
Open [slides.html](file:///d:/Semester%209/IR%20Package/slides.html) in your browser. Use Arrow Keys to navigate, and press `S` to toggle speaker notes for each presenter.

---

## 📐 Mathematical Formulation

### 1. Content-Based Cosine Similarity
$$\text{Sim}(\mathbf{u}, \mathbf{v}_j) = \frac{\mathbf{u} \cdot \mathbf{v}_j}{\|\mathbf{u}\|_2 \|\mathbf{v}_j\|_2}$$
where $\mathbf{u} = \frac{1}{|H_u|} \sum_{i \in H_u} w_i \mathbf{v}_i$.

### 2. Matrix Factorization (SVD)
$$\min_{U, V} \sum_{(u, i) \in R} (r_{ui} - \mathbf{p}_u^T \mathbf{q}_i)^2 + \lambda (\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2)$$

### 3. Hybrid Multi-Criteria Fusion
$$\text{Relevance}(u, i) = \alpha \cdot S_{\text{Content}}(u, i) + \beta \cdot S_{\text{Collab}}(u, i) + \gamma \cdot \text{QualityPrior}(i) + \text{ContextBoost}$$

### 4. Maximal Marginal Relevance (MMR Diversity)
$$\text{MMR} = \arg\max_{d \in \mathcal{C} \setminus \mathcal{S}} \left[ \lambda \cdot \text{Rel}(d) - (1 - \lambda) \max_{d_j \in \mathcal{S}} \text{Sim}(d, d_j) \right]$$
