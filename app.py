import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="H&M Hybrid Recommender", layout="wide", page_icon="🛍️")

@st.cache_data
def load_articles():
    articles_path = "data/articles.csv"
    if os.path.exists(articles_path):
        df = pd.read_csv(articles_path, dtype={'article_id': str})
        
        info_map = {}
        for _, row in df.iterrows():
            aid_str = str(row['article_id']).zfill(10)
            try:
                aid_int = int(row['article_id'])
            except ValueError:
                aid_int = None
                
            details = {
                'group': str(row.get('product_group_name', '')),
                'index': str(row.get('index_name', '')),
                'type': str(row.get('product_type_name', ''))
            }
            
            info_map[aid_str] = details
            if aid_int is not None:
                info_map[aid_int] = details
                
        return info_map
    return {}

article2info = load_articles()

st.title("🛍️ H&M Fashion Recommendation System")
st.markdown("**Ավարտական նախագիծ**՝ *Hybrid Model (Repeat Purchases + Time-Decay ALS)*")

st.divider()

tab1, tab2 = st.tabs(["📊 Model Evaluation", "🎯 Customer Recommendations"])

with tab1:
    st.header("Test Set Performance Metrics")
    results_path = "results/model_test_results.csv"
    
    if os.path.exists(results_path):
        df_results = pd.read_csv(results_path, index_col=0)
        
        st.dataframe(
            df_results.style.highlight_max(axis=0, color="#d4edda"),
            use_container_width=True
        )
        st.info("💡 **Best Model:** Hybrid (5 Repeat + Time-Decay ALS Decay) ապահովում է առավելագույն MAP@12 և NDCG@12:")
    else:
        st.error("`results/model_test_results.csv` ֆայլը չի գտնվել:")

with tab2:
    st.header("Get Personal Recommendations")
    
    recs_path = "results/customer_recommendations.csv"
    
    if os.path.exists(recs_path):
        recs_df = pd.read_csv(recs_path)
        
        sample_customers = recs_df['customer_id'].head(5).tolist()
        st.caption(f"Փորձարկման համար օրինակ ID-ներ՝ `{'6cc121e5cc202d2bf344ffe795002bdbf87178054bcda2e57161f0ef810a4b55'}`")
        
        user_input = st.text_input("Մուտքագրեք Customer ID-ն:", value='6cc121e5cc202d2bf344ffe795002bdbf87178054bcda2e57161f0ef810a4b55')
        
        
        EXCLUDED_KEYWORDS = {
            'underwear', 'socks', 'tights', 'nightwear', 'swimwear', 
            'bra', 'lingerie', 'underwear/nightwear', 'socks & tights'
        }
        
        if st.button("Ստանալ Top-12 Առաջարկները", type="primary"):
            user_recs = recs_df[recs_df['customer_id'] == user_input]
            
            if not user_recs.empty:
                items_str = user_recs.iloc[0]['recommendations']
                items_list = items_str.split()
                
                filtered_items = []
                for item_id in items_list:
                    clean_id_str = str(item_id).strip().zfill(10)
                    try:
                        clean_id_int = int(item_id)
                    except ValueError:
                        clean_id_int = None

                    info = article2info.get(clean_id_str, article2info.get(clean_id_int, {}))
                    
                    grp = info.get('group', '').lower()
                    idx = info.get('index', '').lower()
                    tp = info.get('type', '').lower()
                    
                    
                    is_excluded = any(
                        kw in grp or kw in idx or kw in tp 
                        for kw in EXCLUDED_KEYWORDS
                    )
                    
                    if not is_excluded:
                        filtered_items.append(item_id)
                
                display_items = filtered_items[:12]
                
                st.success(f"Գտնված են 12 առաջարկներ `{user_input[:15]}...`-ի համար․")
                
                cols = st.columns(4)
                for idx_col, item_id in enumerate(display_items):
                    with cols[idx_col % 4]:
                        st.markdown(f"**#{idx_col+1} Article ID**")
                        st.code(item_id, language="text")
                        
                        formatted_item_id = str(item_id).zfill(10)
                        folder_name = formatted_item_id[:3]
                        image_path = f"data/images/{folder_name}/{formatted_item_id}.jpg"

                        if os.path.exists(image_path):
                            st.image(image_path, use_container_width=True)
                        else:
                            st.image(
                                "https://via.placeholder.com/200x250/f0f2f6/333333?text=No+Image", 
                                use_container_width=True
                            )
            else:
                st.warning("Այս Customer ID-ն չի գտնվել test բազայում:")
    else:
        st.error("`results/customer_recommendations.csv` ֆայլը չի գտնվել:")