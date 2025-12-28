# notebooks/data_preprocessing.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_config, prepare_data, clean_text
import pickle
import yaml

def main():
    print("🔧 VERİ ÖN İŞLEME VE HAZIRLAMA")
    print("=" * 60)
    
    # 1. Config yükle
    print("1. Yapılandırma dosyası yükleniyor...")
    config = load_config()
    print(f"   ✅ config.yaml yüklendi")
    
    # 2. Veriyi yükle
    print("\n2. Ham veri yükleniyor...")
    data_path = config['paths']['raw_data']
    
    if not os.path.exists(data_path):
        print(f"   ❌ Hata: {data_path} bulunamadı!")
        return
    
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    print(f"   ✅ {len(df)} örnek yüklendi")
    
    # 3. Veriyi incele
    print("\n3. Veri analizi:")
    print(f"   - Sütunlar: {df.columns.tolist()}")
    print(f"   - Skor aralığı: {df['sentiment_score'].min():.1f} - {df['sentiment_score'].max():.1f}")
    print(f"   - Ortalama skor: {df['sentiment_score'].mean():.2f}")
    print(f"   - Kategori dağılımı:")
    for cat, count in df['category'].value_counts().items():
        print(f"       {cat}: {count} ({count/len(df)*100:.1f}%)")
    
    # 4. Örnek metinleri göster
    print("\n4. Örnek metinler (temizleme öncesi/sonrası):")
    sample_indices = [0, 100, 500]
    for idx in sample_indices:
        original = df.iloc[idx]['text']
        cleaned = clean_text(original)
        print(f"\n   Örnek {idx+1}:")
        print(f"   Orijinal: {original}")
        print(f"   Temizlenmiş: {cleaned}")
        print(f"   Skor: {df.iloc[idx]['sentiment_score']} - Kategori: {df.iloc[idx]['category']}")
    
    # 5. Görselleştirmeler
    print("\n5. Görselleştirmeler oluşturuluyor...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 5.1 Skor dağılımı
    axes[0, 0].hist(df['sentiment_score'], bins=21, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Nötr')
    axes[0, 0].set_xlabel('Duygu Skoru')
    axes[0, 0].set_ylabel('Frekans')
    axes[0, 0].set_title('Duygu Skorları Dağılımı')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 5.2 Kategori dağılımı
    category_counts = df['category'].value_counts().sort_index()
    colors = ['#ff6666', '#ff9999', '#cccccc', '#99ff99', '#00cc00']  # Kırmızıdan yeşile
    axes[0, 1].bar(category_counts.index, category_counts.values, color=colors)
    axes[0, 1].set_xlabel('Kategori')
    axes[0, 1].set_ylabel('Frekans')
    axes[0, 1].set_title('Kategori Dağılımı')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Her çubuğa sayı ekle
    for i, (cat, count) in enumerate(category_counts.items()):
        axes[0, 1].text(i, count + 50, str(count), ha='center', va='bottom', fontweight='bold')
    
    # 5.3 Metin uzunlukları
    df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
    axes[0, 2].hist(df['text_length'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 2].axvline(x=df['text_length'].mean(), color='red', linestyle='--', 
                      label=f'Ortalama: {df["text_length"].mean():.1f}')
    axes[0, 2].set_xlabel('Metin Uzunluğu (kelime)')
    axes[0, 2].set_ylabel('Frekans')
    axes[0, 2].set_title('Metin Uzunlukları Dağılımı')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 5.4 Skor vs Uzunluk scatter
    axes[1, 0].scatter(df['sentiment_score'], df['text_length'], alpha=0.3, s=20, color='purple')
    axes[1, 0].set_xlabel('Duygu Skoru')
    axes[1, 0].set_ylabel('Metin Uzunluğu')
    axes[1, 0].set_title('Skor vs Metin Uzunluğu')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5.5 Zaman içinde skor değişimi
    df['date'] = pd.to_datetime(df['date'])
    df_sorted = df.sort_values('date')
    
    # Aylık ortalama skor
    df_sorted['year_month'] = df_sorted['date'].dt.to_period('M')
    monthly_avg = df_sorted.groupby('year_month')['sentiment_score'].mean().reset_index()
    monthly_avg['year_month'] = monthly_avg['year_month'].astype(str)
    
    axes[1, 1].plot(range(len(monthly_avg)), monthly_avg['sentiment_score'], 
                   marker='o', linewidth=2, color='blue')
    axes[1, 1].set_xlabel('Ay')
    axes[1, 1].set_ylabel('Ortalama Duygu Skoru')
    axes[1, 1].set_title('Zaman İçinde Duygu Değişimi')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    # X eksenini ayarlama (her 3 ayda bir göster)
    tick_positions = range(0, len(monthly_avg), 3)
    tick_labels = [monthly_avg.iloc[i]['year_month'] for i in tick_positions]
    axes[1, 1].set_xticks(tick_positions)
    axes[1, 1].set_xticklabels(tick_labels, rotation=45)
    
    # 5.6 Şirket bazlı skorlar
    company_scores = df.groupby('company')['sentiment_score'].mean().sort_values()
    top_10_companies = pd.concat([company_scores.head(5), company_scores.tail(5)])
    
    colors_bar = ['#ff6666' if score < 0 else '#00cc00' for score in top_10_companies.values]
    axes[1, 2].barh(range(len(top_10_companies)), top_10_companies.values, color=colors_bar)
    axes[1, 2].set_yticks(range(len(top_10_companies)))
    axes[1, 2].set_yticklabels(top_10_companies.index)
    axes[1, 2].set_xlabel('Ortalama Duygu Skoru')
    axes[1, 2].set_title('Şirket Bazlı Ortalama Duygu Skorları\n(En Olumsuz ve En Olumlu 5 Şirket)')
    axes[1, 2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Görselleri kaydet
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(f'{output_dir}/data_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/data_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Görselleştirmeler kaydedildi: {output_dir}/data_analysis.png")
    
    # 6. Veriyi hazırla ve tokenize et
    print("\n6. Veri hazırlanıyor (temizleme, tokenization, split)...")
    processed_data = prepare_data(df, config)
    
    # 7. İşlenmiş verileri kaydet
    print("\n7. İşlenmiş veriler kaydediliyor...")
    
    # processed_data.pickle olarak kaydet
    processed_path = f'{output_dir}/processed_data.pickle'
    with open(processed_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"   ✅ İşlenmiş veriler kaydedildi: {processed_path}")
    
    # 8. İstatistikleri yazdır
    print("\n8. İŞLEME İSTATİSTİKLERİ:")
    print("=" * 60)
    print(f"   Train seti:      {len(processed_data['X_train']):>6} örnek")
    print(f"   Validation seti: {len(processed_data['X_val']):>6} örnek")
    print(f"   Test seti:       {len(processed_data['X_test']):>6} örnek")
    print(f"   Toplam:          {len(processed_data['X_train']) + len(processed_data['X_val']) + len(processed_data['X_test']):>6} örnek")
    print()
    print(f"   Kelime dağarcığı boyutu: {processed_data['vocab_size']}")
    print(f"   Sequence uzunluğu:       {processed_data['X_train'].shape[1]}")
    print(f"   Embedding boyutu:        {config['model']['embedding_dim']}")
    
    # 9. Train setinden örnek göster
    print("\n9. ÖRNEK TOKENIZATION:")
    print("-" * 40)
    
    sample_idx = 42  # Rastgele bir örnek
    tokenizer = processed_data['tokenizer']
    
    # Orijinal metni bul
    original_text = df.iloc[sample_idx]['text']
    cleaned_text = df.iloc[sample_idx]['cleaned_text']
    true_score = processed_data['y_train'][sample_idx] if sample_idx < len(processed_data['y_train']) else "N/A"
    
    # Sequence'i al
    if sample_idx < len(processed_data['X_train']):
        sample_sequence = processed_data['X_train'][sample_idx]
        
        # Sıfır olmayan token'ları al
        non_zero_tokens = sample_sequence[sample_sequence != 0]
        
        print(f"   Örnek Index: {sample_idx}")
        print(f"   Orijinal metin: {original_text}")
        print(f"   Temizlenmiş: {cleaned_text}")
        print(f"   Gerçek skor: {true_score}")
        print(f"   Token sayısı: {len(non_zero_tokens)}")
        print(f"   Token IDs (ilk 10): {non_zero_tokens[:10].tolist()}")
        
        # ID'leri kelimelere çevir
        if hasattr(tokenizer, 'index_word'):
            words = [tokenizer.index_word.get(token_id, f'[UNK_{token_id}]') 
                    for token_id in non_zero_tokens[:10]]
            print(f"   Kelimeler (ilk 10): {words}")
    
    print("\n" + "=" * 60)
    print("🎉 VERİ ÖN İŞLEME BAŞARIYLA TAMAMLANDI!")
    print("=" * 60)
    print("\n📁 OLUŞTURULAN DOSYALAR:")
    print(f"   1. data/processed/processed_data.pickle")
    print(f"   2. models/tokenizer.pickle")
    print(f"   3. data/processed/data_analysis.png")
    print(f"   4. data/processed/data_analysis.pdf")
    
    # 10. Model için hazırlık bilgileri
    print("\n🔧 MODEL EĞİTİMİ İÇİN HAZIRLIK:")
    print("-" * 40)
    print(f"   Batch size:        {config['training']['batch_size']}")
    print(f"   Epoch sayısı:      {config['training']['epochs']}")
    print(f"   Learning rate:     {config['training']['learning_rate']}")
    print(f"   Embedding boyutu:  {config['model']['embedding_dim']}")
    print(f"   FastText kullanımı: {'EVET' if config['model']['use_fasttext'] else 'HAYIR'}")

if __name__ == "__main__":
    main()