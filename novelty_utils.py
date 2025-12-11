import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity

def load_embedding_file(path):
    """
    Parquet embedding dosyasını okur ve:
    - publishDate parse eder
    - embedding JSON -> list -> numpy array dönüştürür
    - tarih sırasına göre sıralar
    """
    df = pd.read_parquet(path)

    # publishDate normalize
    df["publishDate"] = pd.to_datetime(df["publishDate"], dayfirst=True, errors="coerce")

    # embedding JSON -> numpy
    df["embedding"] = df["embedding"].apply(lambda x: np.array(json.loads(x)))

    # Tarih sırasına göre sırala
    df = df.sort_values("publishDate").reset_index(drop=True)

    return df


def compute_v_history(embeddings, idx):
    """
    Leakage-free centroid.
    idx = şu anki event
    history = 0..idx-1
    """
    if idx == 0:
        return None

    past_vectors = embeddings[:idx]
    return np.mean(np.stack(past_vectors), axis=0)


def novelty_score(v_new, v_history):
    """
    novelty = 1 - cosine_similarity
    """
    if v_history is None:
        return 0.0

    sim = cosine_similarity(
        v_new.reshape(1, -1),
        v_history.reshape(1, -1)
    )[0][0]

    return float(1 - sim)
