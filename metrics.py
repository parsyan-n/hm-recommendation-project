import pandas as pd 
import numpy as np 

articles=pd.read_parquet('data/articles_processed.parquet')
customers=pd.read_parquet('data/customers_processed.parquet')
transactions=pd.read_parquet('data/transactions_processed.parquet')


last_date=transactions['t_dat'].max()
val_start=last_date-pd.Timedelta(days=6)
train=transactions[transactions['t_dat']<val_start]
val=transactions[transactions['t_dat']>=val_start]

#================================================================================
#                      Metrics
#================================================================================

def evaluate(predictions, actuals, total_catalog_items, novelty_dict, k=12):
    precisions, recalls, map_scores, novelty_scores, ndcg_scores = [], [], [], [], []
    recommended_unique_items = set()

    #discounts vector for NDCG@K speedup
    discounts = np.log2(np.arange(2, k + 2))

    for user_id, recs in predictions.items():
        if user_id not in actuals:
            continue

        true_items = set(actuals[user_id])
        if len(true_items) == 0:
            continue

        top_k_recs = recs[:k]

        # 1. Precision & Recall
        hits = len(set(top_k_recs) & true_items)
        precisions.append(hits / k)
        recalls.append(hits / len(true_items))

        # 2. Average Precision (MAP@K)
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(top_k_recs):
            if p in true_items and p not in top_k_recs[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        map_scores.append(score / min(len(true_items), k))

        # 3. NDCG@K
        hits_binary = [1 if p in true_items else 0 for p in top_k_recs]
        if any(hits_binary):
            dcg = np.sum(hits_binary / discounts[:len(hits_binary)])
            ideal_hits = sorted(hits_binary, reverse=True)
            idcg = np.sum(ideal_hits / discounts[:len(ideal_hits)])
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

        # 4. Novelty (Self-Information)
        user_novelty = np.mean([novelty_dict.get(item, 20.0) for item in top_k_recs])
        novelty_scores.append(user_novelty)

        # 5. Coverage
        recommended_unique_items.update(top_k_recs)

    return {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        f"MAP@{k}": np.mean(map_scores),
        f"NDCG@{k}": np.mean(ndcg_scores),
        f"Novelty@{k}": np.mean(novelty_scores),
        "Coverage": len(recommended_unique_items) / total_catalog_items,
    }