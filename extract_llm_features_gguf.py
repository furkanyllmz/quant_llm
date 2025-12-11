import os
import json
import glob
import argparse
import requests
import time
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ENV (stability)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ======================================================
# AYARLAR
# ======================================================
LLAMA_SERVER_URL = "http://127.0.0.1:8080"


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
# LLAMA.CPP SERVER FUNCTIONS
# ======================================================
def check_server(server_url: str) -> bool:
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def build_chat_prompt(system_prompt: str, user_content: str) -> str:
    im_start = "<" + "|im_start|" + ">"
    im_end = "<" + "|im_end|" + ">"
    
    prompt = f"{im_start}system\n{system_prompt}{im_end}\n{im_start}user\n{user_content}{im_end}\n{im_start}assistant\n"
    return prompt


def run_llama_inference(prompt: str, server_url: str, max_tokens: int = 256) -> str:
    im_end = "<" + "|im_end|" + ">"
    endoftext = "<" + "|endoftext|" + ">"
    
    try:
        response = requests.post(
            f"{server_url}/completion",
            json={
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": 0.1,
                "stop": [im_end, endoftext],
                "stream": False
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("content", "").strip()
        else:
            print(f"   ⚠️ Server hatası: {response.status_code}")
            return ""
    
    except requests.exceptions.ConnectionError:
        print("   ❌ llama-server'a bağlanılamadı! Server çalışıyor mu?")
        return ""
    except requests.exceptions.Timeout:
        print("   ⚠️ Timeout")
        return ""
    except Exception as e:
        print(f"   ⚠️ API hatası: {e}")
        return ""


# ======================================================
# MAIN FEATURE EXTRACTION PER SYMBOL
# ======================================================
def extract_features_for_symbol(
    symbol: str,
    system_prompt: str,
    context_dir: str,
    output_dir: str,
    server_url: str,
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

        # Prompt oluştur
        prompt = build_chat_prompt(system_prompt, text)

        try:
            out_text = run_llama_inference(
                prompt=prompt,
                server_url=server_url,
                max_tokens=256
            )
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
    parser.add_argument("--prompt-path", "-p", default="./system_prompt.txt")
    parser.add_argument("--server", "-s", default=LLAMA_SERVER_URL)
    parser.add_argument("--symbol", "-sym", default=None)
    parser.add_argument("--max-records", "-n", type=int, default=None)
    args = parser.parse_args()

    # Server kontrolü
    print(f"🔧 llama-server: {args.server}")
    if not check_server(args.server):
        print(f"\n❌ llama-server çalışmıyor!")
        print(f"\nÖnce şu komutu çalıştır:")
        print(f"  ./llama-server.exe -m qwen-kap-final-Q4_K_M.gguf -ngl 35 -c 4096 --port 8080")
        return
    
    print("✅ Server bağlantısı OK\n")

    os.makedirs(args.output_dir, exist_ok=True)

    print("🔄 SYSTEM PROMPT yükleniyor...")
    system_prompt = load_system_prompt(args.prompt_path)
    print("✅ SYSTEM PROMPT yüklendi.\n")

    if args.symbol:
        symbols = [args.symbol]
    else:
        files = glob.glob(os.path.join(args.context_dir, "*.json"))
        symbols = [os.path.basename(f).replace(".json", "") for f in files]

    for sym in symbols:
        extract_features_for_symbol(
            symbol=sym,
            system_prompt=system_prompt,
            context_dir=args.context_dir,
            output_dir=args.output_dir,
            server_url=args.server,
            max_records=args.max_records,
        )


if __name__ == "__main__":
    main()
