import io
import os
import threading

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from remotezip import RemoteZip

ZIP_URL = "https://huggingface.co/datasets/Nonna123321/hm_images/resolve/main/images.zip"
RESULTS_PATH = "results/model_test_results.csv"
RECS_PATH = "results/customer_recommendations.csv"

# Example customer: one of the 20 most active customers in the train period (heavy buyer)
HEAVY_BUYER_ID = "6cc121e5cc202d2bf344ffe795002bdbf87178054bcda2e57161f0ef810a4b55"

N_DISPLAY = 12
ACCURACY_COLS = ["MAP@12", "NDCG@12", "Precision@12", "Recall@12", "Coverage"]

st.set_page_config(page_title="H&M Hybrid Recommender", layout="wide", page_icon="🛍️")


# ------------------------------------------------------------------------------
# Data loading (cached)
# ------------------------------------------------------------------------------
@st.cache_data
def load_recommendations():
    """customer_id -> space-separated article ids (top-12, already filtered)."""
    if not os.path.exists(RECS_PATH):
        return None
    df = pd.read_csv(RECS_PATH, dtype={"customer_id": str})
    return df.set_index("customer_id")["recommendations"]


@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    df = pd.read_csv(RESULTS_PATH, index_col=0)
    if "Model" in df.columns:
        df = df.set_index("Model")
    return df


# ------------------------------------------------------------------------------
# Images 
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_remote_zip():
    return RemoteZip(ZIP_URL), threading.Lock()


@st.cache_data(show_spinner=False)
def fetch_image_bytes(item_id):
    """Returns the jpg bytes, or None if the image is not in the archive.
    Raises on network errors, so failures are not cached."""
    item = str(item_id).strip().zfill(10)
    folder = item[:3]
    candidates = [f"images/{folder}/{item}.jpg", f"{folder}/{item}.jpg"]

    last_error = None
    for _ in range(2):  
        try:
            rzip, lock = get_remote_zip()
            with lock:
                for path in candidates:
                    try:
                        return rzip.read(path)
                    except KeyError:
                        continue
            return None
        except Exception as e:
            last_error = e
            get_remote_zip.clear()
    raise RuntimeError(f"Could not fetch image {item}") from last_error


def load_image(item_id):
    try:
        data = fetch_image_bytes(item_id)
    except Exception:
        return None
    if data is None:
        return None
    try:
        return Image.open(io.BytesIO(data))
    except Exception:
        return None


def placeholder_image(width=200, height=250):
    img = Image.new("RGB", (width, height), "#f0f2f6")
    ImageDraw.Draw(img).text((width // 2 - 30, height // 2 - 5), "No image", fill="#555555")
    return img


# ------------------------------------------------------------------------------
# Page
# ------------------------------------------------------------------------------
st.title("🛍️ H&M Fashion Recommendation System")
st.markdown("**Ավարտական նախագիծ**՝ *Hybrid Model (Repeat Purchases + Item-Item Cosine)*")

with st.expander("ℹ️ Մոդելի և demo-ի մասին"):
    st.markdown(
        """
* **Մոդել.** հաճախորդի վերջին 4 շաբաթվա 6 ամենահաճախ գնված ապրանքները դրվում են ցուցակի սկզբում, մնացածը լրացվում է Item-Item Cosine մոդելի առաջարկներով։
* **Առաջարկները նախապես հաշվված են** և վերաբերում են միայն այն հաճախորդներին, ովքեր գնում են կատարել test շաբաթվա ընթացքում։
"""
    )

st.divider()

tab1, tab2 = st.tabs(["📊 Model Evaluation", "🎯 Customer Recommendations"])

# ------------------------------------------------------------------------------
# Tab 1: evaluation table
# ------------------------------------------------------------------------------
with tab1:
    st.header("Test Set Performance Metrics")
    df_results = load_results()

    if df_results is not None:
        numeric_cols = df_results.select_dtypes("number").columns.tolist()
        highlight_cols = [c for c in ACCURACY_COLS if c in numeric_cols]
        styled = (
            df_results.style
            .highlight_max(subset=highlight_cols, axis=0, color="#d4edda")
            .format({c: "{:.5f}" for c in numeric_cols})
        )
        st.dataframe(styled)
        st.info(
            "💡 **Final Model:** Hybrid (6 Repeat + Item-Item Cosine). Ընտրված է validation-ում "
            "MAP@12-ի և Coverage-ի հավասարակշռությամբ (Time-Decay ALS hybrid-ն ուներ ավելի բարձր MAP, "
            "բայց Coverage-ը շատ ցածր էր)։"
        )
    else:
        st.error(f"`{RESULTS_PATH}` ֆայլը չի գտնվել:")

# ------------------------------------------------------------------------------
# Tab 2: recommendations
# ------------------------------------------------------------------------------
with tab2:
    st.header("Get Personal Recommendations")
    recs = load_recommendations()

    if recs is None:
        st.error(f"`{RECS_PATH}` ֆայլը չի գտնվել:")
    else:
        if "customer_id_input" not in st.session_state:
            st.session_state.customer_id_input = HEAVY_BUYER_ID

        col_a, col_b = st.columns(2)
        if col_a.button("🎲 Պատահական հաճախորդ"):
            st.session_state.customer_id_input = recs.sample(1).index[0]
        if col_b.button("⭐ Ակտիվ գնորդի օրինակ"):
            st.session_state.customer_id_input = HEAVY_BUYER_ID

        user_input = st.text_input("Մուտքագրեք Customer ID-ն:", key="customer_id_input").strip()

        if st.button("Ստանալ Top-12 առաջարկները", type="primary"):
            if user_input in recs.index:
                display_items = recs[user_input].split()[:N_DISPLAY]

                st.success(f"{len(display_items)} առաջարկ՝ `{user_input[:15]}...` հաճախորդի համար")

                cols = st.columns(4)
                with st.spinner("Նկարները բեռնվում են..."):
                    for rank, item_id in enumerate(display_items, start=1):
                        with cols[(rank - 1) % 4]:
                            img = load_image(item_id)
                            st.image(img if img is not None else placeholder_image(),
                                     caption=f"#{rank} · {item_id}")
            else:
                st.warning(
                    "Այս Customer ID-ն չկա ցուցակում. առաջարկները գեներացված են միայն այն "
                    "հաճախորդների համար, ովքեր գնում են կատարել test շաբաթվա ընթացքում։"
                )