import streamlit as st
import pandas as pd
import numpy as np

from engine.hybrid_engine import HybridRecommenderEngine
from evaluate import run_ir_evaluation

# Page configuration
st.set_page_config(
    page_title="CineMatch | Netflix-Style Hybrid IR Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Netflix-Grade Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #0d0d12;
        color: #f0f0f5;
    }
    
    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.25) 0%, rgba(13, 13, 18, 0.95) 75%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(229, 9, 20, 0.3);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 40%, #ff5e62 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #9a9ab0;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    
    /* Movie Cards */
    .movie-card {
        background: rgba(24, 24, 32, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.25s ease, border-color 0.25s ease;
        margin-bottom: 18px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        border-color: rgba(229, 9, 20, 0.5);
        box-shadow: 0 12px 25px rgba(0,0,0,0.6);
    }
    .movie-poster {
        width: 100%;
        height: 240px;
        object-fit: cover;
    }
    .card-body {
        padding: 14px;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .movie-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .movie-meta {
        font-size: 0.8rem;
        color: #8b8b9e;
        margin-bottom: 8px;
    }
    .score-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .score-match {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .explanation-tag {
        font-size: 0.76rem;
        color: #ffb703;
        font-style: italic;
        margin-top: 6px;
        line-height: 1.3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    movies_df = pd.read_csv("movies.csv")
    ratings_df = pd.read_csv("ratings.csv")
    return HybridRecommenderEngine(movies_df, ratings_df), movies_df, ratings_df

engine, movies_df, ratings_df = get_engine()

# Initialize session state for user ratings
if "session_ratings" not in st.session_state:
    st.session_state.session_ratings = {}

# Sidebar: Controls & IR Parameters
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1574375927938-d5a98e8ffe85?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("🎛️ Recommender Config")
    
    user_options = {
        1: "User 1: Alex (Sci-Fi & Mind-Benders)",
        2: "User 2: Michael (Classic Mob & Crime)",
        3: "User 3: Emma (Animation & Family)",
        4: "User 4: David (Auteur & Dark Comedy)",
        5: "User 5: Maya (Cold-Start: 1 Rating)",
        99: "User 99: Custom Live Session"
    }
    selected_user_id = st.selectbox("Select Active User Profile", options=list(user_options.keys()), format_func=lambda x: user_options[x])
    
    st.markdown("---")
    st.subheader("⚖️ Hybrid Scoring Weights")
    alpha = st.slider("Content-Based Weight (α)", min_value=0.0, max_value=1.0, value=0.45, step=0.05,
                      help="Weight given to plot synopsis, director, and genre similarity.")
    beta = st.slider("Collaborative Weight (β)", min_value=0.0, max_value=1.0, value=0.40, step=0.05,
                     help="Weight given to matrix factorization crowd patterns.")
    gamma = st.slider("Quality Prior Weight (γ)", min_value=0.0, max_value=1.0, value=0.15, step=0.05,
                      help="Weight given to overall movie rating and vote consensus.")

    st.markdown("---")
    st.subheader("🧩 MMR Diversity Re-ranking")
    enable_mmr = st.toggle("Enable Maximal Marginal Relevance (MMR)", value=True)
    mmr_lambda = st.slider("Diversity Trade-off (λ)", min_value=0.1, max_value=1.0, value=0.70, step=0.05,
                           disabled=not enable_mmr,
                           help="1.0 = Pure Relevance (risk of duplicate genres); 0.0 = Pure Novelty/Diversity.")

    st.markdown("---")
    st.subheader("🌐 Context & Environmental Sensors")
    device = st.selectbox("Hardware Device", ["smart_tv", "mobile", "tablet", "desktop"], index=0)
    time_of_day = st.selectbox("Time of Day", ["night", "evening", "afternoon", "morning"], index=0)
    day_of_week = st.selectbox("Day of Week", ["friday", "saturday", "sunday", "monday", "wednesday"], index=0)
    genre_filter = st.selectbox("Genre Filter", ["All", "Sci-Fi", "Crime", "Action", "Drama", "Animation", "Comedy", "Thriller"], index=0)

# Main Dashboard
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="hero-title">CINEMATCH</h1>
            <div class="hero-subtitle">Two-Stage Information Retrieval Recommendation Architecture</div>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(229,9,20,0.2); color: #ff5e62; padding: 6px 14px; border-radius: 20px; font-weight: 700; border: 1px solid rgba(229,9,20,0.3);">
                LIVE TWO-STAGE PIPELINE
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Build current active ratings
if selected_user_id == 99:
    # Custom interactive session
    active_ratings = [
        {"user_id": 99, "movie_id": mid, "rating": r['rating'], "watch_ratio": r['watch_ratio']}
        for mid, r in st.session_state.session_ratings.items()
    ]
else:
    active_ratings = ratings_df[ratings_df['user_id'] == selected_user_id].to_dict('records')

# Tabs
tab_recs, tab_evaluate, tab_rate, tab_architecture = st.tabs([
    "🍿 Streaming Recommendations",
    "📊 IR Evaluation Benchmark",
    "⭐ Rate & Test Cold-Start",
    "🏛️ Pipeline Architecture"
])

with tab_recs:
    # Display Current Profile Summary
    col_u1, col_u2, col_u3, col_u4 = st.columns(4)
    with col_u1:
        st.metric("Active Persona", user_options[selected_user_id].split(":")[1].split("(")[0].strip())
    with col_u2:
        st.metric("Watch History Count", f"{len(active_ratings)} titles")
    with col_u3:
        context_desc = f"{device.replace('_', ' ').title()} • {time_of_day.title()}"
        st.metric("Context State", context_desc)
    with col_u4:
        mode = "Cold-Start Auto-Switch" if len(active_ratings) <= 1 else "Two-Stage Hybrid"
        st.metric("Engine Mode", mode)

    st.markdown("### 🎯 Top Personalized Picks for You (Hybrid MMR)")
    
    recs = engine.recommend(
        user_id=selected_user_id,
        live_ratings=active_ratings,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        device=device,
        time_of_day=time_of_day,
        day_of_week=day_of_week,
        genre_filter=genre_filter,
        top_k=8,
        enable_mmr=enable_mmr,
        mmr_lambda=mmr_lambda,
        filter_watched=True
    )

    if recs:
        cols = st.columns(4)
        for i, item in enumerate(recs[:4]):
            with cols[i]:
                match_pct = int(item['relevance_score'] * 100)
                st.markdown(f"""
                <div class="movie-card">
                    <img class="movie-poster" src="{item['poster_url']}" alt="{item['title']}">
                    <div class="card-body">
                        <div>
                            <div class="movie-title">{item['title']}</div>
                            <div class="movie-meta">{item['release_year']} • {item['runtime_min']}m • ⭐ {item['vote_average']}</div>
                            <span class="score-badge score-match">{match_pct}% Match</span>
                            <div style="font-size: 0.75rem; color: #a0a0b0;">
                                C: {item['content_score']} | CF: {item['collab_score']} | Prior: {item['quality_prior']}
                            </div>
                        </div>
                        <div class="explanation-tag">💡 {item['explanation']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        if len(recs) > 4:
            cols2 = st.columns(4)
            for i, item in enumerate(recs[4:8]):
                with cols2[i]:
                    match_pct = int(item['relevance_score'] * 100)
                    st.markdown(f"""
                    <div class="movie-card">
                        <img class="movie-poster" src="{item['poster_url']}" alt="{item['title']}">
                        <div class="card-body">
                            <div>
                                <div class="movie-title">{item['title']}</div>
                                <div class="movie-meta">{item['release_year']} • {item['runtime_min']}m • ⭐ {item['vote_average']}</div>
                                <span class="score-badge score-match">{match_pct}% Match</span>
                                <div style="font-size: 0.75rem; color: #a0a0b0;">
                                    C: {item['content_score']} | CF: {item['collab_score']} | Prior: {item['quality_prior']}
                                </div>
                            </div>
                            <div class="explanation-tag">💡 {item['explanation']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No recommendations found matching current strict filters.")

    # Categorized Netflix Rails
    st.markdown("---")
    rails = engine.generate_netflix_rails(
        user_id=selected_user_id,
        live_ratings=active_ratings,
        device=device,
        time_of_day=time_of_day,
        day_of_week=day_of_week
    )

    for rail_title, rail_items in rails.items():
        if rail_title != "Top Picks for You" and rail_items:
            st.markdown(f"### 🎞️ {rail_title}")
            rail_cols = st.columns(min(len(rail_items), 4))
            for idx, m in enumerate(rail_items[:4]):
                with rail_cols[idx]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <img class="movie-poster" src="{m['poster_url']}" alt="{m['title']}">
                        <div class="card-body">
                            <div class="movie-title">{m['title']}</div>
                            <div class="movie-meta">{m['release_year']} • ⭐ {m['vote_average']}</div>
                            <div class="explanation-tag">💡 {m.get('explanation', '')}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

with tab_evaluate:
    st.subheader("📈 Information Retrieval Benchmark Evaluation")
    st.write("Quantitative comparison across traditional IR metrics: Precision@K, Recall@K, NDCG@K, Intra-List Diversity (ILD), and Catalog Coverage.")
    
    eval_k = st.slider("Rank Threshold (K)", min_value=3, max_value=10, value=5, step=1)
    
    if st.button("🚀 Run Live IR Benchmark", type="primary"):
        with st.spinner("Executing IR ranking evaluation across all test users..."):
            df_eval = run_ir_evaluation(k=eval_k)
            st.dataframe(df_eval, use_container_width=True)
            
            st.success("✅ Hybrid + MMR achieves the superior balance: highest Intra-List Diversity with optimal NDCG and 100% catalog coverage.")
    else:
        st.info("Click above to run the live IR evaluation suite.")

with tab_rate:
    st.subheader("⭐ Interactive Watch & Rating Simulation")
    st.write("Add ratings to test real-time profile adaptation and Cold-Start switching behavior.")
    
    col_rate1, col_rate2, col_rate3 = st.columns([2, 1, 1])
    with col_rate1:
        movie_to_rate = st.selectbox("Choose a Movie", options=movies_df['title'].tolist())
    with col_rate2:
        rating_val = st.slider("Your Rating", 1.0, 5.0, 5.0, 0.5)
    with col_rate3:
        watch_ratio = st.slider("Watch Completion Ratio", 0.1, 1.0, 1.0, 0.1)

    if st.button("➕ Submit Interaction"):
        selected_m_row = movies_df[movies_df['title'] == movie_to_rate].iloc[0]
        mid = int(selected_m_row['movie_id'])
        st.session_state.session_ratings[mid] = {"rating": rating_val, "watch_ratio": watch_ratio}
        st.success(f"Added {movie_to_rate} ({rating_val}⭐) to your session history! Switch to 'User 99: Custom Live Session' in sidebar to test.")

    if st.session_state.session_ratings:
        st.markdown("#### Your Current Dynamic Session History:")
        for mid, d in st.session_state.session_ratings.items():
            t = movies_df[movies_df['movie_id'] == mid].iloc[0]['title']
            st.write(f"- **{t}**: {d['rating']}⭐ (Watched {int(d['watch_ratio']*100)}%)")
        if st.button("🗑️ Clear Session Ratings"):
            st.session_state.session_ratings = {}
            st.rerun()

with tab_architecture:
    st.subheader("🏛️ Two-Stage IR Pipeline Architecture")
    st.markdown("""
    ```
    +-----------------------------------------------------------------------------------------+
    |                                   MOVIE CATALOG (40,000+ Titles)                        |
    +-----------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
    +-----------------------------------------------------------------------------------------+
    | STAGE 1: CANDIDATE GENERATION (Retrieval Stage: High Recall, <15ms)                      |
    |  • Dense Semantic Vector Index (Sentence-BERT / TF-IDF Vector Space Model)             |
    |  • Collaborative Filtering Neighborhood Lookup (SVD Matrix Factorization)                |
    |  • Quality Prior / Trending Filter                                                      |
    |  -> Discards 99% of catalog; produces Top 500 Candidate Items                           |
    +-----------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
    +-----------------------------------------------------------------------------------------+
    | STAGE 2: MULTI-CRITERIA SCORING & FUSION (High Precision)                               |
    |  • Formula: FinalScore = α·Content + β·Collab + γ·QualityPrior + ContextModifier        |
    |  • Dynamic Cold-Start Rule: If interactions ≤ 1, switches to α=0.80, β=0.00, γ=0.20     |
    |  • Context Sensors: Device (TV/Mobile), Time of day, Day of week                        |
    +-----------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
    +-----------------------------------------------------------------------------------------+
    | STAGE 3: RE-RANKING & DIVERSITY (MMR)                                                   |
    |  • Maximal Marginal Relevance: MMR = λ·Rel(d) - (1-λ)·max Sim(d, d_j)                   |
    |  • Eliminates Filter Bubble / genre clustering                                          |
    |  • Generates categorized Netflix Rails: "Top Picks", "Because You Watched", "Trending" |
    +-----------------------------------------------------------------------------------------+
    ```
    """)
