import os
import glob
import pandas as pd

# ============================
# CONFIG
# ============================
MASTER_PATH = "./master_df_test.csv"
NOVELTY_DIR = "./novelty"
LLM_DIR = "./llm_features"
OUTPUT_PATH = "./merged_master_test.csv"

# Sadece 4 hisseyi kullanıyoruz
TARGET_SYMBOLS = ["SASA", "HEKTS", "THYAO", "TUPRS"]


# ============================
# HELPERS
# ============================
def load_novelty():
    rows = []
    files = glob.glob(os.path.join(NOVELTY_DIR, "*_novelty.parquet"))
    for fp in files:
        sym = os.path.basename(fp).replace("_novelty.parquet", "")
        if sym not in TARGET_SYMBOLS:
            continue
        df = pd.read_parquet(fp)
        df["symbol"] = sym
        df.rename(columns={"publishDate": "date"}, inplace=True)
        rows.append(df[["symbol", "date", "novelty"]])
    return pd.concat(rows, ignore_index=True)


def load_llm():
    rows = []
    files = glob.glob(os.path.join(LLM_DIR, "*_llm_features.parquet"))
    for fp in files:
        sym = os.path.basename(fp).replace("_llm_features.parquet", "")
        if sym not in TARGET_SYMBOLS:
            continue
        df = pd.read_parquet(fp)
        df["symbol"] = sym
        df.rename(columns={"publishDate": "date"}, inplace=True)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ============================
# MAIN MERGE LOGIC
# ============================
def main():

    print("📌 master_df_test yükleniyor...")
    master = pd.read_csv(MASTER_PATH)
    master["date"] = pd.to_datetime(master["date"])
    master = master[master["symbol"].isin(TARGET_SYMBOLS)]

    print("📌 Novelty yükleniyor...")
    df_novelty = load_novelty()
    df_novelty["date"] = pd.to_datetime(df_novelty["date"])

    print("📌 LLM features yükleniyor...")
    df_llm = load_llm()
    df_llm["date"] = pd.to_datetime(df_llm["date"])

    print("🔄 Merge 1: master + novelty")
    df = master.merge(df_novelty, on=["symbol", "date"], how="left")

    print("🔄 Merge 2: master + llm_features")
    df = df.merge(df_llm, on=["symbol", "date"], how="left")

    # Eksik olan KAP olmayan günler → doldur
    llm_cols = [
        "fine_sentiment","risk_score","impact_score","tone_shift",
        "uncertainty_score","actionability","financial_impact_strength",
        "forward_vs_backward_ratio","regulatory_pressure"
    ]

    print("🧹 Null dolduruluyor (LLM + Novelty olmayan günler)")
    df["novelty"] = df["novelty"].fillna(0)

    for c in llm_cols:
        df[c] = df[c].fillna(0)

    df["shock_direction"] = df["shock_direction"].fillna("neutral")
    df["reasoning"] = df["reasoning"].fillna("")

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"💾 Kaydediliyor → {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)

    print("🎉 Tamamlandı! Birleşik dataset hazır.")


if __name__ == "__main__":
    main()
