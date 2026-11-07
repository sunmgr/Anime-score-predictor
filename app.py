import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Anime Score Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Glassmorphism & Modern Anime Aesthetic CSS
st.markdown(
    """
    <style>
    /* Main Background with Dark Aesthetic Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Target main containers */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Glassmorphism Card Style */
    div.css-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 25px;
    }

    /* Custom Header Styling */
    .title-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #c084fc, #6366f1, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Streamlit Input Widget Label Styling */
    label, .stMultiSelect label, .stSelectbox label, .stNumberInput label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Custom Primary Button Styling */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #8b5cf6 0%, #ec4899 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4);
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.6);
        color: #ffffff;
    }

    /* Styled Result Card */
    .result-box {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%);
        border: 1px solid rgba(236, 72, 153, 0.4);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        backdrop-filter: blur(10px);
        margin-top: 20px;
    }

    .score-number {
        font-size: 3.5rem;
        font-weight: 900;
        color: #f472b6;
        text-shadow: 0 0 20px rgba(244, 114, 182, 0.5);
    }

    .score-label {
        font-size: 1.1rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Asset Loading (Cached for performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("optimized_rf_model.pkl")
    mlb = joblib.load("genre_binarizer.pkl")
    model_features = joblib.load("model_features.pkl")
    return model, mlb, model_features

try:
    model, mlb, model_features = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Header Section
# -----------------------------------------------------------------------------
st.markdown(
    '<h1 class="title-header">✨ Anime Rating Predictor</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle-header">Predict your anime\'s MyAnimeList (MAL) score using machine learning</p>',
    unsafe_allow_html=True,
)

# Sidebar Info Panel
with st.sidebar:
    st.title("About")
    st.info(
        "This tool uses a Random Forest Regressor trained on MyAnimeList data "
        "to estimate an anime's reception based on metadata, release year, and genres."
    )
    st.divider()
    st.caption("Powered by Scikit-Learn & Streamlit")

# -----------------------------------------------------------------------------
# 4. Input Form Section
# -----------------------------------------------------------------------------
st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.subheader("⚙️ Anime Attributes")

col1, col2 = st.columns(2)

with col1:
    episodes = st.number_input(
        "Number of Episodes", min_value=1, value=12, step=1
    )
    released_year = st.number_input(
        "Release Year", min_value=1960, max_value=2030, value=2024, step=1
    )
    anime_type = st.selectbox(
        "Type", ["TV", "Movie", "OVA", "ONA", "Special", "Music"]
    )

with col2:
    source = st.selectbox(
        "Source Material",
        ["Manga", "Original", "Light Novel", "Visual Novel", "Other"],
    )
    rating = st.selectbox(
        "Age Rating",
        [
            "G - All Ages",
            "PG - Children",
            "PG-13 - Teens 13 or older",
            "R - 17+ (violence & profanity)",
            "R+ - Mild Nudity",
        ],
    )

selected_genres = st.multiselect(
    "Select Genres",
    options=list(mlb.classes_),
    placeholder="Choose genres (e.g., Action, Romance, Sci-Fi)",
)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Prediction Logic & Output
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Estimate MAL Rating"):
    with st.spinner("Analyzing parameters with Random Forest model..."):
        # 1. Transform genres
        genre_encoded = mlb.transform([selected_genres])
        genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

        # 2. Assemble baseline structure
        input_data = pd.DataFrame({
            "Episodes": [episodes],
            "Released_Year": [released_year],
            "Type": [anime_type],
            "Source": [source],
            "Rating": [rating],
        })

        # 3. Encode categorical variables safely
        input_combined = pd.concat([input_data, genre_df], axis=1)
        input_encoded = pd.get_dummies(
            input_combined, columns=["Type", "Source", "Rating"]
        )

        # 4. Reindex to strictly match expected model feature order
        final_input = input_encoded.reindex(
            columns=model_features, fill_value=0
        )

        # 5. Model Prediction
        predicted_score = float(model.predict(final_input)[0])

    # Display Results in a glassmorphic card
    st.markdown(
        f"""
        <div class="result-box">
            <div class="score-label">Predicted MyAnimeList Score</div>
            <div class="score-number">{predicted_score:.2f} <span style="font-size: 1.5rem; color: #94a3b8;">/ 10</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )