import glob
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from novelty_utils import load_embedding_file, compute_v_history, novelty_score


EMBEDDING_DIR = "./embeddings"
OUTPUT_DIR = "./novelty"

os.makedirs(OUTPUT_DIR, exist_ok=True)

embedding_files = glob.glob(f"{EMBEDDING_DIR}/*_embeddings.parquet")

print(f"{len(embedding_files)} sembol bulundu.\n")

for file_path in embedding_files:
    symbol = os.path.basename(file_path).replace("_embeddings.parquet", "")
    print(f"📄 İşleniyor: {symbol}")

    df = load_embedding_file(file_path)

    # embeddings listesi
    vectors = list(df["embedding"])

    novelty_list = []

    for i in range(len(df)):
        v_new = vectors[i]
        v_hist = compute_v_history(vectors, i)
        n_score = novelty_score(v_new, v_hist)
        novelty_list.append(n_score)

    df_out = pd.DataFrame({
        "symbol": df["symbol"],
        "publishDate": df["publishDate"],
        "novelty": novelty_list
    })

    out_path = f"{OUTPUT_DIR}/{symbol}_novelty.parquet"
    pq.write_table(pa.Table.from_pandas(df_out), out_path)

    print(f"   ✔ Kaydedildi: {out_path}")

print("\n🎉 Tüm novelty hesaplaması tamamlandı!")
