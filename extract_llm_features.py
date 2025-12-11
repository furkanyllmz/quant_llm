import os
import json
import glob
import argparse
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ======================================================
# ENV (Mac stability)
# ======================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ======================================================
# SYSTEM PROMPT LOADER
# ======================================================
def load_system_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"SYSTEM PROMPT dosyası bulunamadı: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ======================================================
# JSON LOADER
# ======================================================
def load_json_records(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # JSON / JSONL / multi-block fallback
    try:
        obj = json.loads(content)
        return obj if isinstance(obj, list) else [obj]
    except:
        pass

    blocks = content.split("}\n{")
    if len(blocks) > 1:
        for i, block in enumerate(blocks):
            if i == 0:
                block += "}"
            elif i == len(blocks) - 1:
                block = "{" + block
            else:
                block = "{" + block + "}"
            try:
                records.append(json.loads(block))
            except:
                pass
        return records

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except:
            pass

    return records


# ======================================================
# MODEL OUTPUT PARSER
# ======================================================
def parse_llm_json(text: str):
    if not text:
        return None

    s = text.find("{")
    e = text.rfind("}") + 1
    if s == -1 or e <= s:
        return None

    block = text[s:e]
    try:
        return json.loads(block)
    except:
        return None


# ======================================================
# MAIN FEATURE EXTRACTION PER SYMBOL
# ======================================================
def extract_features_for_symbol(
    symbol: str,
    model,
    tokenizer,
    system_prompt: str,
    context_dir: str,
    output_dir: str,
    max_records: int = None,
):
    json_path = os.path.join(context_dir, f"{symbol}.json")
    if not os.path.exists(json_path):
        print(f"⚠ {symbol}: JSON bulunamadı, atlanıyor.")
        return

    out_path = os.path.join(output_dir, f"{symbol}_llm_features.parquet")
    if os.path.exists(out_path):
        print(f"⏭ {symbol}: zaten işlenmiş, atlanıyor.")
        return

    records = load_json_records(json_path)
    if not records:
        print(f"⚠ {symbol}: kayıt yok, atlanıyor.")
        return

    print(f"📄 {symbol}: {len(records)} kayıt işleniyor...")
    rows = []
    count = 0

    for rec in tqdm(records, desc=f"LLM {symbol}"):

        if max_records and count >= max_records:
            break

        context = rec.get("context", "")
        if not context:
            continue

        text = context[:4000]  # güvenli sınır

        raw_date = rec.get("publishDate", "")

        try:
            publish_date = datetime.strptime(raw_date, "%d.%m.%Y %H:%M:%S")
        except:
            publish_date = raw_date

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        try:
            out_text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=256,
                sampler=make_sampler(temp=0.1),
                verbose=False,
            ).strip()
        except Exception as e:
            print(f"⚠ Generate error: {e}")
            continue

        parsed = parse_llm_json(out_text)
        if not parsed:
            continue

        row = {
            "symbol": symbol,
            "publishDate": publish_date,

            "fine_sentiment": float(parsed.get("fine_sentiment", 0)),
            "risk_score": float(parsed.get("risk_score", 0)),
            "impact_score": float(parsed.get("impact_score", 0)),
            "tone_shift": float(parsed.get("tone_shift", 0)),
            "uncertainty_score": float(parsed.get("uncertainty_score", 0)),
            "actionability": float(parsed.get("actionability", 0)),
            "financial_impact_strength": float(parsed.get("financial_impact_strength", 0)),
            "forward_vs_backward_ratio": float(parsed.get("forward_vs_backward_ratio", 0)),
            "regulatory_pressure": float(parsed.get("regulatory_pressure", 0)),
            "shock_direction": str(parsed.get("shock_direction", "neutral")),
            "reasoning": str(parsed.get("reasoning", "")),

        }

        rows.append(row)
        count += 1

    if not rows:
        print(f"⚠ {symbol}: Çıktı yok, kaydedilmedi.")
        return

    df = pd.DataFrame(rows)
    df["publishDate"] = pd.to_datetime(df["publishDate"], errors="coerce", dayfirst=True)
    df = df.sort_values("publishDate")

    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path)

    print(f"✅ {symbol}: {out_path} — {len(df)} satır")


# ======================================================
# MAIN
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-dir", "-c", default="./context_data")
    parser.add_argument("--output-dir", "-o", default="./llm_features")
    parser.add_argument("--model-path", "-m", default="./qwen-kap-final")
    parser.add_argument("--prompt-path", "-p", default="./system_prompt.txt")
    parser.add_argument("--symbol", "-s", default=None)
    parser.add_argument("--max-records", "-n", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("🔄 SYSTEM PROMPT yükleniyor...")
    system_prompt = load_system_prompt(args.prompt_path)
    print("✅ SYSTEM PROMPT yüklendi.\n")

    print("🔄 Model yükleniyor...")
    model, tokenizer = load(args.model_path)
    print("✅ Model yüklendi.\n")

    if args.symbol:
        symbols = [args.symbol]
    else:
        files = glob.glob(os.path.join(args.context_dir, "*.json"))
        symbols = [os.path.basename(f).replace(".json", "") for f in files]

    for sym in symbols:
        extract_features_for_symbol(
            symbol=sym,
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            context_dir=args.context_dir,
            output_dir=args.output_dir,
            max_records=args.max_records,
        )


if __name__ == "__main__":
    main()
