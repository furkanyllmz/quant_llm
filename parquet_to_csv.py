#!/usr/bin/env python3
"""
Parquet dosyalarını CSV formatına dönüştüren script
"""

import pandas as pd
import glob
import os
from pathlib import Path


def convert_parquet_to_csv(input_dir, output_dir=None, pattern="*_features.parquet"):
    """
    Belirtilen dizindeki tüm parquet dosyalarını CSV'ye dönüştür
    
    Args:
        input_dir (str): Parquet dosyalarının bulunduğu dizin
        output_dir (str): CSV dosyalarının kaydedileceği dizin (varsayılan: input_dir)
        pattern (str): Dosya adı pattern'i (varsayılan: "*_features.parquet")
    
    Returns:
        dict: Dönüştürülen dosyaların bilgisi
    """
    
    if output_dir is None:
        output_dir = input_dir
    
    # Çıktı dizinini oluştur
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Parquet dosyalarını bul
    parquet_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    
    if not parquet_files:
        print(f"⚠️  '{input_dir}' dizininde '{pattern}' pattern'i eşleştiren dosya bulunamadı!")
        return {}
    
    print(f"📊 {len(parquet_files)} parquet dosyası bulundu.\n")
    print(f"{'='*80}")
    
    results = {}
    total_rows = 0
    total_files = 0
    
    for parquet_path in parquet_files:
        try:
            # Dosya adını al
            filename = os.path.basename(parquet_path)
            
            # Parquet dosyasını oku
            df = pd.read_parquet(parquet_path)
            
            # CSV yolunu oluştur
            csv_filename = filename.replace('.parquet', '.csv')
            csv_path = os.path.join(output_dir, csv_filename)
            
            # CSV'ye kaydet
            df.to_csv(csv_path, index=False)
            
            # Bilgiler
            file_size = os.path.getsize(csv_path)
            rows, cols = df.shape
            
            results[filename] = {
                'parquet_path': parquet_path,
                'csv_path': csv_path,
                'rows': rows,
                'columns': cols,
                'csv_size_kb': file_size / 1024,
                'status': '✅ Başarılı'
            }
            
            total_rows += rows
            total_files += 1
            
            print(f"✅ {csv_filename}")
            print(f"   Satır: {rows:,} | Sütun: {cols}")
            print(f"   Boyut: {file_size/1024:.2f} KB")
            print(f"   Çıktı: {csv_path}")
            print()
            
        except Exception as e:
            results[filename] = {
                'status': f'❌ Hata: {str(e)}'
            }
            print(f"❌ {filename}")
            print(f"   Hata: {str(e)}")
            print()
    
    print(f"{'='*80}")
    print(f"\n📈 ÖZETİ:")
    print(f"   Başarılı: {total_files} dosya")
    print(f"   Toplam satır: {total_rows:,}")
    print(f"   Çıktı dizini: {output_dir}")
    
    return results


def convert_novelty_parquets(input_dir="./novelty", output_dir="./novelty_csv"):
    """
    Novelty parquet dosyalarını CSV'ye dönüştür
    """
    print("\n🔄 NOVELTY PARQUETLERINI CSV'YE DÖNÜŞTÜRÜYORUM...\n")
    return convert_parquet_to_csv(input_dir, output_dir, pattern="*_novelty.parquet")


def convert_llm_features_parquets(input_dir="./llm_features", output_dir="./llm_features"):
    """
    LLM Features parquet dosyalarını CSV'ye dönüştür
    """
    print("\n🔄 LLM FEATURES PARQUETLERINI CSV'YE DÖNÜŞTÜRÜYORUM...\n")
    return convert_parquet_to_csv(input_dir, output_dir, pattern="*_llm_features.parquet")


def main():
    """
    Ana fonksiyon - tüm parquetleri dönüştür
    """
    print("="*80)
    print("🔄 PARQUET → CSV DÖNÜŞTÜRÜCÜ")
    print("="*80)
    
    # LLM Features dönüştür
    llm_results = convert_llm_features_parquets()
    
    # Novelty dönüştür (eğer dizin varsa)
    if os.path.exists("./novelty"):
        novelty_results = convert_novelty_parquets()
    
    print("\n🎉 Tüm dönüştürmeler tamamlandı!")


if __name__ == "__main__":
    main()
