# H&M Fashion Recommendation System

Ավարտական նախագիծ (Armenian Code Academy, ML Engineer track). Hybrid
recommendation համակարգ H&M-ի իրական գնումների տվյալների վրա (31M+
transaction, 1.3M+ հաճախորդ, 105K+ ապրանք)՝ EDA-ից մինչև deployed
Streamlit demo։

**🔗 Live demo:** https://hm-recommendation-project-nj7wtwcj9fznj9upxrpyat.streamlit.app/

## Նպատակը

Խնդիրը personalized recommendation է՝ յուրաքանչյուր հաճախորդի համար
կանխատեսել, թե մոտակա շաբաթվա ընթացքում ինչ 12 ապրանք կգնի, հիմնվելով
նրա նախկին գնումների և ապրանքների նմանության վրա։

## Dataset

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
(Kaggle)՝ transactions, articles (ապրանքների մետատվյալներ), customers,
և ապրանքների նկարներ։ Տվյալները մեծ են (~31M տող), ուստի repo-ում
commit չեն արված (տես `.gitignore`)։

**Ինչպես ներբեռնել.**
1. Kaggle-ից ներբեռնիր dataset-ը (պահանջում է Kaggle հաշիվ)։
2. Տեղադրիր `data/` թղթապանակում, հետևյալ կառուցվածքով.
 

 data/
├── transactions_train.csv
├── articles.csv
├── customers.csv


## Methodology

1. **EDA & Feature Engineering** (`EDA_FeatureEngineering.ipynb`) — տվյալների
   հետազոտում, missing value-ների մշակում, feature-ների ստեղծում։
2. **Train/Val/Test split** — վերջին երկու շաբաթը train-ից առանձնացված
   է. նախավերջին շաբաթը՝ validation, վերջինը՝ test։
3. **Baselines** (`Baselines.ipynb`) — 4 պարզ baseline (global popularity,
   recent popularity, repeat purchases + fallback), գնահատված val-ի վրա։
4. **Experimentation** (`validation_experiments.ipynb`) — Item-Item Cosine
   Similarity, ALS matrix factorization (grid search-ով tuned), ALS
   time-decay կշիռներով, և hybrid մոդելներ (վերջին գնումներ + collaborative
   filtering fallback)։ Բոլոր hyperparameter-ների ընտրությունը արված է
   բացառապես val-ի վրա։
5. **Final test evaluation** (`final_test_evaluation.ipynb`) — վերջնական
   մոդելը, ընտրված val-ով, գնահատված test-ի վրա **մեկ անգամ**։

## Վերջնական մոդել

**Hybrid: 6 recent repeat purchases + Item-Item Cosine Similarity**

Հաճախորդի վերջին 4 շաբաթվա 6 ամենահաճախ գնված ապրանքները դրվում են
ցուցակի սկզբում, մնացած տեղերը լրացվում են Cosine similarity-ի
առաջարկություններով։

**Ընտրության հիմնավորումը.** Val-ի արդյունքներով ALS time-decay հիբրիդն
ուներ ավելի բարձր MAP@12 (0.0256 vs 0.0211), բայց Coverage-ն ընդամենը
0.12 էր (6 անգամ ավելի ցածր, քան Cosine-հիբրիդի 0.68-ը)՝ նշանակում է՝
գրեթե միշտ նույն մի քանի «հանրաճանաչ» ապրանքն էր առաջարկում։ Ընտրեցինք
Cosine-հիբրիդը՝ MAP-ի և Coverage-ի միջև ավելի առողջ հավասարակշռության
համար։

### Արդյունքներ (Test set)

| Model | MAP@12 | NDCG@12 | Precision@12 | Recall@12 | Coverage |
|---|---|---|---|---|---|
| Global Popularity | 0.00290 | 0.00531 | 0.00198 | 0.00775 | 0.00011 |
| Recent Popularity | 0.00875 | 0.01576 | 0.00591 | 0.02550 | 0.00011 |
| Repeat + Popularity | 0.00970 | 0.01538 | 0.00444 | 0.02459 | 0.56390 |
| Top 5 Repeat + Popularity | 0.01056 | 0.01831 | 0.00614 | 0.03143 | 0.42387 |
| **Final: 6 Repeat + Cosine** | **0.02163** | **0.03010** | **0.00732** | **0.04069** | **0.67524** |

Ամբողջական աղյուսակը՝ [`results/model_test_results.csv`](results/model_test_results.csv)։

## Repo կառուցվածքը

├── app.py # Streamlit demo
├── metrics.py # Evaluation metrics (MAP, NDCG, Precision, Recall, Novelty, Coverage)
├── EDA_FeatureEngineering.ipynb
├── Baselines.ipynb
├── validation_experiments.ipynb
├── final_test_evaluation.ipynb
├── requirements.txt 
├── Dockerfile
├── data/ # (git-ignored, ինքդ ներբեռնիր Kaggle-ից)
└── results/ # Պահպանված metric-ներ և recommendation-ներ


## Ինչպես աշխատացնել

### Demo (առանց local տեղադրման)
Live demo link-ը՝ վերևում։

### Local-ում
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Docker-ով
```bash
docker build -t hm-recommender .
docker run -p 8080:8080 hm-recommender
```
Հետո բացիր `http://localhost:8080`։

### Notebook-ները կրկին աշխատացնելու համար
Պահանջվում են լրացուցիչ փաթեթներ (numpy, scikit-learn, implicit,
huggingface_hub, requests)՝ արդեն requirements.txt-ում ներառված են։
Նախ ներբեռնիր dataset-ը , հետո աշխատացրու
notebook-երը հերթականությամբ՝ EDA → Baselines → validation_experiments
→ final_test_evaluation։

## Հայտնի սահմանափակումներ

- Cold-start հաճախորդները (առանց գնումների պատմության) ստանում են
  միայն popularity-based fallback, ոչ թե personalized առաջարկներ։
- Ապրանքների նկարներից (CV embeddings) content-based signal չի
  օգտագործվել մոդելում. demo-ում նկարները միայն ցուցադրական նպատակով
  են բեռնվում remote zip-ից։
- Coverage-ի ու MAP-ի միջև trade-off-ը նշանակում է, որ ցուցակի tail-ում
  գտնվող ապրանքների մեծ մասը դեռ հազվադեպ են երևում։