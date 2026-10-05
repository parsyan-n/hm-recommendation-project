import numpy as np


def evaluate(predictions, actuals, total_catalog_items, novelty_dict, k=12):
    precisions, recalls, map_scores, novelty_scores, ndcg_scores = [], [], [], [], []
    recommended_unique_items = set()

    # position 1..k -> log2(2)..log2(k+1)
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

        # 2. MAP@K
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(top_k_recs):
            if p in true_items and p not in top_k_recs[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        map_scores.append(score / min(len(true_items), k))

        # 3. NDCG@K
        hits_binary = np.array([1 if p in true_items else 0 for p in top_k_recs])
        dcg = np.sum(hits_binary / discounts[:len(hits_binary)])
        n_ideal = min(len(true_items), k)               # best case: all real purchases at the top
        idcg = np.sum(1.0 / discounts[:n_ideal])
        ndcg_scores.append(dcg / idcg)

        # 4. Novelty (self-information)
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