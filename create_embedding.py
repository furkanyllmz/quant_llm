import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import glob
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
from FlagEmbedding import BGEM3FlagModel

INPUT_DIR = "./context_data"
OUTPUT_DIR = "./embeddings"
MODEL_NAME = "BAAI/bge-m3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading {MODEL_NAME} ...")
model = BGEM3FlagModel(
    MODEL_NAME,
    use_fp16=False,
    device="cpu"
)
print("Model loaded.\n")

def load_json_records(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    lines = content.split("}\n{")
    if len(lines) > 1:
        for i, line in enumerate(lines):
            if i == 0: line += "}"
            elif i == len(lines)-1: line = "{" + line
            else: line = "{" + line + "}"
            try: recs.append(json.loads(line))
            except: pass
    else:
        try:
            obj = json.loads(content)
            recs = obj if isinstance(obj, list) else [obj]
        except:
            for line in content.split("\n"):
                try: recs.append(json.loads(line))
                except: pass
    return recs

json_files = glob.glob(f"{INPUT_DIR}/*.json")
print(f"{len(json_files)} files found.\n")

for json_file in json_files:
    symbol = os.path.basename(json_file).replace(".json", "")
    out_path = f"{OUTPUT_DIR}/{symbol}_embeddings.parquet"

    if os.path.exists(out_path):
        print(f"⏭ {symbol} exists, skipping.")
        continue

    records = load_json_records(json_file)
    rows = []

    for rec in tqdm(records, desc=symbol):
        text = rec.get("context", "")
        if not text:
            continue

        text = text[:2048]

        result = model.encode(
            [text],
            batch_size=1,
            max_length=2048,
            return_dense=True,
            return_sparse=False
        )
        vec = result["dense_vecs"][0]

        # ❗ Embedding'i JSON string olarak kaydediyoruz → bozulmaz
        rows.append({
            "symbol": symbol,
            "publishDate": rec.get("publishDate", ""),
            "embedding": json.dumps(vec.tolist())
        })

    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df), out_path)

    print(f"✔ Saved: {out_path}\n")

print("DONE.")
