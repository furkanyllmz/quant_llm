import os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier


# =========================================
# CONFIG
# =========================================
DATA_PATH = "./merged_master_test.csv"

# Hedef:
#   future_ret_10d = close.shift(-10) / close - 1
#   y_event_10d = 1 if future_ret_10d > 0.02 else 0
HORIZON = 10
THRESHOLD = 0.02  # +%2


# =========================================
# LOAD & BASIC CLEAN
# =========================================
def load_data():
    print(f"📂 Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


# =========================================
# TARGET ENGINEERING
# =========================================
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Gelecek 10 günlük getiri (aynı hisse içinde)
    df["future_ret_10d"] = (
        df.groupby("symbol")["price_close"]
        .apply(lambda s: s.shift(-HORIZON) / s - 1.0)
        .reset_index(level=0, drop=True)
    )

    # Binary target
    df["y_event_10d"] = (df["future_ret_10d"] > THRESHOLD).astype(int)

    # Geleceği bilinmeyen son HORIZON satırı uçsun
    df = df[~df["future_ret_10d"].isna()].copy()

    return df


# =========================================
# FEATURE SET HAZIRLAMA
# =========================================
def prepare_features(df: pd.DataFrame):
    # shock_direction'ı numerik ek feature'a çevir
    shock_map = {
        "up": 1,
        "down": -1,
        "neutral": 0,
        "": 0,
        None: 0
    }
    df["shock_dir_num"] = df["shock_direction"].map(shock_map).fillna(0)

    # has_kap zaten merge scriptinde var
    if "has_kap" not in df.columns:
        df["has_kap"] = (df["novelty"] > 0).astype(int)

    # Hangi kolonlar kesinlikle feature DEĞİL?
    drop_exact = [
        "date",
        "reasoning",
        "future_ret_10d",
        "y_event_10d",
        "period",  # '2020/3' gibi string format, feature olarak kullanılamaz
        "announcement_date",  # datetime
        "publishDate",  # datetime
    ]

    # Her ihtimale karşı target / label olabilecek kolonları da at
    drop_prefixes = ("future_ret_", "y_event_", "label_", "target_")

    cols_to_drop = set(drop_exact)
    for c in df.columns:
        if any(c.startswith(p) for p in drop_prefixes):
            cols_to_drop.add(c)

    # Kategorik kolonları otomatik tespit et (object dtype olanlar)
    # Ama drop edilecek, target ve datetime kolonları hariç tut
    datetime_like_cols = ["date", "period", "announcement_date", "publishDate"]
    all_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [
        c for c in all_object_cols 
        if c not in cols_to_drop and c not in ["date", "future_ret_10d", "y_event_10d"] and c not in datetime_like_cols
    ]
    
    # Kategorik kolonları string'e çevir (CatBoost gereksinimi)
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("missing")
    
    print(f"🏷️  Kategorik kolonlar: {cat_cols}")

    # shock_direction'ı numerik ek feature'a çevir (zaten var)
    # Ama kategorik olarak da kullanacağız

    # Son feature listesi (symbol ve diğer kategorikler dahil)
    feature_cols = [
        c for c in df.columns
        if c not in cols_to_drop and c not in ["date", "future_ret_10d", "y_event_10d"]
    ]

    # df_model oluştururken sadece gerekli kolonları al
    # feature_cols zaten symbol içeriyor, tekrar eklemeyelim
    df_model = df[["date", "future_ret_10d", "y_event_10d"] + feature_cols].copy()

    return df_model, feature_cols, cat_cols


# =========================================
# TRAIN / VALID SPLIT (TIME-BASED)
# =========================================
def train_valid_split(df_model: pd.DataFrame):
    # Tarihe göre %80 train, %20 valid
    cutoff = df_model["date"].quantile(0.8)
    train = df_model[df_model["date"] <= cutoff].copy()
    valid = df_model[df_model["date"] > cutoff].copy()

    print(f"🧱 Train range: {train['date'].min().date()} → {train['date'].max().date()}  (n={len(train)})")
    print(f"🧪 Valid range: {valid['date'].min().date()} → {valid['date'].max().date()}  (n={len(valid)})")

    return train, valid


# =========================================
# MODEL TRAIN
# =========================================
def train_model(train, valid, feature_cols, cat_cols):
    X_train = train[feature_cols].copy()
    y_train = train["y_event_10d"].copy()

    X_valid = valid[feature_cols].copy()
    y_valid = valid["y_event_10d"].copy()

    # CatBoost için kategorik kolon index'leri
    cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.03,
        iterations=800,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_indices
    )

    return model


# =========================================
# EVALUATION: AUC, LIFT, IC
# =========================================
def evaluate_model(model, valid, feature_cols):
    X_valid = valid[feature_cols]
    y_valid = valid["y_event_10d"].values
    ret_valid = valid["future_ret_10d"].values

    proba = model.predict_proba(X_valid)[:, 1]

    # AUC
    auc = roc_auc_score(y_valid, proba)
    print(f"\n📈 AUC (y_event_10d): {auc:.4f}")

    # Lift @ Top10%
    n = len(valid)
    k = max(1, int(n * 0.10))
    top_idx = np.argsort(proba)[::-1][:k]

    avg_ret_top = ret_valid[top_idx].mean()
    avg_ret_all = ret_valid.mean()

    lift = avg_ret_top / avg_ret_all if avg_ret_all != 0 else np.nan
    print(f"💹 Avg future_ret_10d (all):  {avg_ret_all:.4f}")
    print(f"💹 Avg future_ret_10d (top10%): {avg_ret_top:.4f}")
    print(f"🚀 Lift@10%: {lift:.2f}x")

    # IC (Information Coefficient)
    ic = np.corrcoef(proba, ret_valid)[0, 1]
    print(f"🧠 IC (pred vs future_ret_10d): {ic:.4f}")

    return {
        "auc": auc,
        "lift10": lift,
        "ic": ic
    }


# =========================================
# MAIN
# =========================================
def main():
    df = load_data()
    df = add_targets(df)

    df_model, feature_cols, cat_cols = prepare_features(df)
    train, valid = train_valid_split(df_model)

    print(f"🔢 Feature count: {len(feature_cols)}")
    print("🧾 Feature örneği:", feature_cols[:10])

    model = train_model(train, valid, feature_cols, cat_cols)

    metrics = evaluate_model(model, valid, feature_cols)

    # İstersen modeli kaydedebilirsin
    model.save_model("catboost_nlp_alpha_10d.cbm")
    print("\n💾 Model saved: catboost_nlp_alpha_10d.cbm")


if __name__ == "__main__":
    main()
