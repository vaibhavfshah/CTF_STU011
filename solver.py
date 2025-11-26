import pandas as pd
import numpy as np
import re
import hashlib
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


df_book = pd.read_csv("filtered_book_reviews.csv") 

def clean_text(t): # to remove html tags and extra spaces
    t = str(t).lower()
    t = re.sub(r"<br\s*/?>", " ", t)
    t = re.sub(r"</?[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

df_book["clean"] = df_book["text"].apply(clean_text)

# ---- Weak labeling rules ----
superlatives = [
    "amazing","best","awesome","fantastic","highly recommend",
    "love it","excellent","must read","great book"
]

def suspicion_rule(row):
    text = row["clean"]
    S_short = int(len(text) < 120)
    S_long = int(len(text) > 500) * -1
    S_super = int(any(w in text for w in superlatives))
    num_sent = text.count(".")
    num_commas = text.count(",")
    S_simple = int(num_sent <= 1 or num_commas == 0)
    S_votes = int(row["helpful_vote"] == 0 and len(text) < 200)

    score = (4*S_super) + (3*S_short) + (3*S_simple) + (2*S_votes) - (3*S_long)
    return int(score >= 3)

df_book["label"] = df_book.apply(suspicion_rule, axis=1)

# Train model
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df_book["clean"])
y = df_book["label"]

model = LogisticRegression(max_iter=500)
model.fit(X, y)

# Score all reviews
df_book["model_score"] = model.predict_proba(X)[:, 1]

# Select genuine reviews for SHAP
threshold = df_book["model_score"].quantile(0.30)
df_genuine = df_book[(df_book["model_score"] <= threshold) & (df_book["clean"].str.len() > 120)]

X_genuine = vectorizer.transform(df_genuine["clean"])

# SHAP
explainer = shap.LinearExplainer(model, X, feature_names=vectorizer.get_feature_names_out())
shap_values = explainer.shap_values(X_genuine)
mean_shap = shap_values.mean(axis=0)

top3_idx = np.argsort(mean_shap)[:3]
top3_words = vectorizer.get_feature_names_out()[top3_idx]

print("Top authenticity words →", list(top3_words))

# Build FLAG3
concat = "".join(top3_words) + "011"
flag3 = hashlib.sha256(concat.encode()).hexdigest()[:10]

print("\nFLAG3= " + flag3)
