# src/data_generator.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import re
import yaml
import sys
import os

# Python path'ine proje kökünü ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# utils modülünden load_config fonksiyonunu al
try:
    from src.utils import load_config
    print("✅ utils.py başarıyla import edildi")
except ImportError as e:
    print(f"⚠️  utils.py import hatası: {e}")
    # Fallback: config'i direkt yükle
    def load_config(config_path="config.yaml"):
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config
        else:
            # Varsayılan config
            print("⚠️  config.yaml bulunamadı, varsayılan config kullanılıyor")
            return {
                'data': {'num_samples': 2000},
                'paths': {'raw_data': 'data/samples.csv'}
            }

class FinancialDataGenerator:
    """Finansal duygu analizi için sentetik veri üretici"""
    
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        random.seed(42)
        np.random.seed(42)
        
        # Finansal terimler sözlüğü
        self.financial_terms = {
            'positive': [
                'yükseliş', 'artış', 'kazanç', 'kar', 'büyüme', 'olumlu',
                'güçlü', 'sağlam', 'başarılı', 'iyi', 'rekor', 'müthiş',
                'harika', 'pozitif', 'gelişme', 'ilerleme', 'fayda', 'getiri',
                'kârlı', 'verimli', 'istikrarlı', 'güvenli', 'cazip', 'parlak'
            ],
            'negative': [
                'düşüş', 'kayıp', 'zarar', 'başarısız', 'kötü', 'zayıf',
                'olumsuz', 'düşük', 'kriz', 'risk', 'korku', 'panik',
                'kaygı', 'endişe', 'çöküş', 'negatif', 'kayıp', 'zarar',
                'tehlikeli', 'istikrarsız', 'belirsiz', 'tedirgin', 'baskı'
            ],
            'companies': [
                'THYAO', 'AKBNK', 'GARAN', 'ISCTR', 'YKBNK', 'ASELS',
                'TAVHL', 'BIMAS', 'MGROS', 'SAHOL', 'KCHOL', 'SASA',
                'PETKM', 'TUPRS', 'FROTO', 'TCELL', 'EREGL', 'TOASO',
                'ARCLK', 'HEKTS', 'ENKAI', 'KORDS', 'VAKBN', 'GUBRF'
            ],
            'sectors': [
                'banka', 'havayolu', 'savunma', 'perakende', 'otomotiv',
                'enerji', 'kimya', 'telekom', 'çelik', 'gayrimenkul',
                'teknoloji', 'sağlık', 'ulaşım', 'gıda', 'inşaat'
            ],
            'verbs': [
                'açıkladı', 'bildirdi', 'duyurdu', 'ilan etti', 'paylaştı',
                'raporladı', 'sunuldu', 'belirtildi', 'ifade edildi',
                'açıklandı', 'yayınlandı', 'iletildi', 'bildirildi'
            ],
            'nouns': [
                'kar', 'ciro', 'satış', 'büyüme', 'performans', 'sonuç',
                'rapor', 'veri', 'istatistik', 'analiz', 'tahmin', 'beklenti',
                'projeksiyon', 'öngörü', 'değerlendirme', 'inceleme'
            ]
        }
        
        # Template'ler farklı duygu skorları için
        self.templates = self._create_templates()
    
    def _create_templates(self):
        """Duygu skorlarına göre template'ler oluştur"""
        return {
            'very_positive': [
                "{company} hissesi {positive} bir {noun} {verb}! Yatırımcılar mutlu.",
                "Şirketin {positive} {noun} açıklaması piyasayı hareketlendirdi.",
                "{company} için {positive} haberler geliyor, fiyatlar tırmanıyor.",
                "Analistler {company} hissesine güçlü alım önerisi verdi.",
                "{sector} sektöründe {positive} gelişmeler yaşanıyor.",
                "{company} {positive} {noun} rakamlarıyla dikkat çekti.",
                "Yatırımcılar {company} hissesinde {positive} hareket bekliyor.",
                "{company} {verb} {positive} bir {noun} performansı sergiledi."
            ],
            'positive': [
                "{company} hissesinde {positive} yönde hareketler gözleniyor.",
                "Şirketin {noun} performansı {positive} olarak değerlendiriliyor.",
                "{company} için {positive} sinyaller alınıyor.",
                "{sector} sektörü {positive} bir seyir izliyor.",
                "{company} {verb} {positive} {noun} verileri.",
                "Piyasada {company} hissesine yönelik {positive} hava hakim.",
                "{company} hissesi {positive} bir trend içinde."
            ],
            'neutral': [
                "{company} hissesi normal seyirde ilerliyor.",
                "Şirket beklentileri karşıladı, piyasa tepkisiz.",
                "{company} hissesinde önemli bir hareket yok.",
                "{sector} sektöründe dengeli bir seyir hakim.",
                "{company} {verb} beklenen {noun} rakamlarını.",
                "Piyasa {company} hissesini izlemeye devam ediyor.",
                "{company} hissesi teknik analizde nötr bölgede."
            ],
            'negative': [
                "{company} hissesinde {negative} yönde gelişmeler var.",
                "Şirketin {noun} performansı {negative} olarak değerlendirildi.",
                "{company} için {negative} riskler görülüyor.",
                "{sector} sektöründe {negative} hava hakim.",
                "{company} {verb} {negative} {noun} verileri.",
                "Piyasada {company} hissesine yönelik {negative} beklentiler var.",
                "{company} hissesi {negative} bir trende girdi."
            ],
            'very_negative': [
                "{company} hissesi {negative} bir {noun} {verb}! Yatırımcılar endişeli.",
                "Şirketten gelen {negative} haberler piyasayı sarstı.",
                "{company} hissesi için alarm zilleri çalıyor.",
                "{sector} sektöründe kriz sinyalleri artıyor.",
                "{company} {negative} {noun} rakamlarıyla şok etti.",
                "Yatırımcılar {company} hissesinden hızla çıkış yapıyor.",
                "{company} {verb} {negative} bir {noun} performansı."
            ]
        }
    
    def _get_sentiment_category(self, score):
        """Skora göre duygu kategorisi belirle"""
        if score >= 7:
            return 'very_positive'
        elif score >= 3:
            return 'positive'
        elif score >= -2:
            return 'neutral'
        elif score >= -6:
            return 'negative'
        else:
            return 'very_negative'
    
    def _add_variations(self, text, score):
        """Metne çeşitlilik ekle (hashtag, emoji, vs.)"""
        variations = []
        
        # Hashtag ekle (%40 ihtimal)
        if random.random() < 0.4:
            hashtags = ['#borsa', '#yatırım', '#hisse', '#finans', '#ekonomi', 
                       '#bist', '#piyasa', '#analiz', '#trade', '#para']
            text += " " + random.choice(hashtags)
            variations.append("hashtag")
        
        # Emoji ekle (%30 ihtimal)
        if random.random() < 0.3:
            if score > 5:
                emojis = ["🚀", "📈", "💹", "💰", "🎯", "⭐"]
                text = random.choice(emojis) + " " + text
            elif score < -5:
                emojis = ["📉", "💥", "🔥", "⚠️", "🔻", "😱"]
                text = random.choice(emojis) + " " + text
            else:
                emojis = ["📊", "📋", "📰", "ℹ️", "🔍", "👁️"]
                text = random.choice(emojis) + " " + text
            variations.append("emoji")
        
        # Kısaltma ekle (%20 ihtimal)
        if random.random() < 0.2:
            abbreviations = ["FYI", "IMO", "BTW", "TLDR", "FWIW", "YTD"]
            text += " (" + random.choice(abbreviations) + ")"
            variations.append("abbreviation")
        
        # Yazım hatası ekle (%10 ihtimal)
        if random.random() < 0.1:
            # Basit bir yazım hatası simülasyonu
            if len(text) > 20:
                pos = random.randint(5, len(text)-5)
                text = text[:pos] + text[pos+1] + text[pos] + text[pos+2:]
                variations.append("typo")
        
        return text, variations
    
    def generate_text(self, sentiment_score):
        """Duygu skoruna göre metin üret"""
        category = self._get_sentiment_category(sentiment_score)
        
        # Template seç
        template = random.choice(self.templates[category])
        
        # Yer tutucuları doldur
        replacements = {
            'company': random.choice(self.financial_terms['companies']),
            'sector': random.choice(self.financial_terms['sectors']),
            'verb': random.choice(self.financial_terms['verbs']),
            'noun': random.choice(self.financial_terms['nouns']),
            'positive': random.choice(self.financial_terms['positive']),
            'negative': random.choice(self.financial_terms['negative'])
        }
        
        # Template'i doldur
        text = template.format(**replacements)
        
        # Çeşitlilik ekle
        text, variations = self._add_variations(text, sentiment_score)
        
        return text
    
    def generate_dataset(self, num_samples=None):
        """Sentetik dataset oluştur"""
        if num_samples is None:
            num_samples = self.config['data']['num_samples']
        
        print(f"📊 {num_samples} adet sentetik finansal veri üretiliyor...")
        print("=" * 60)
        
        data = []
        progress_step = max(1, num_samples // 10)  # Her %10'da bir ilerleme göster
        
        for i in range(num_samples):
            # Daha gerçekçi dağılım için normal dağılım kullan
            # Ortalama 0, standart sapma 4 ile normal dağılım
            raw_score = np.random.normal(0, 4)
            
            # -10 ile +10 arasına kırp ve yuvarla
            sentiment_score = np.clip(raw_score, -10, 10)
            sentiment_score = round(sentiment_score, 1)
            
            # Metin üret
            text = self.generate_text(sentiment_score)
            
            # Tarih oluştur (son 2 yıl içinde)
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2023, 12, 31)
            random_date = start_date + timedelta(days=random.randint(0, 729))
            
            # Kategori belirle
            if sentiment_score >= 7:
                category = "Çok Olumlu"
            elif sentiment_score >= 3:
                category = "Olumlu"
            elif sentiment_score >= -2:
                category = "Nötr"
            elif sentiment_score >= -6:
                category = "Olumsuz"
            else:
                category = "Çok Olumsuz"
            
            # Veriyi ekle
            data.append({
                'id': f"FIN_{i+1:06d}",
                'text': text,
                'sentiment_score': sentiment_score,
                'category': category,
                'date': random_date.strftime('%Y-%m-%d'),
                'time': random_date.strftime('%H:%M:%S'),
                'source': 'synthetic',
                'company': text.split()[0] if text.split() else 'UNKNOWN'
            })
            
            # İlerleme çubuğu
            if (i + 1) % progress_step == 0:
                progress = (i + 1) / num_samples * 100
                print(f"  ██████████████████ {progress:.0f}% tamamlandı ({i + 1}/{num_samples})")
        
        # DataFrame oluştur
        df = pd.DataFrame(data)
        
        print("\n" + "=" * 60)
        print("✅ Veri üretimi tamamlandı!")
        print(f"\n📈 İstatistikler:")
        print(f"   Toplam örnek: {len(df)}")
        print(f"   Skor aralığı: {df['sentiment_score'].min():.1f} - {df['sentiment_score'].max():.1f}")
        print(f"   Ortalama skor: {df['sentiment_score'].mean():.2f}")
        print(f"   Standart sapma: {df['sentiment_score'].std():.2f}")
        
        # Kategori dağılımı
        print("\n🎯 Kategori Dağılımı:")
        category_dist = df['category'].value_counts().sort_index()
        for cat, count in category_dist.items():
            percentage = count / len(df) * 100
            print(f"   {cat:<15}: {count:>4} ({percentage:5.1f}%)")
        
        return df
    
    def save_dataset(self, df, filepath=None):
        """Dataset'i kaydet"""
        if filepath is None:
            filepath = self.config['paths']['raw_data']
        
        # Klasörü oluştur
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # CSV olarak kaydet
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n💾 Veri kaydedildi: {filepath}")
        print(f"   Dosya boyutu: {os.path.getsize(filepath) / 1024:.1f} KB")
        
        return filepath
    
    def analyze_dataset(self, df):
        """Dataset'i analiz et ve raporla"""
        print("\n📊 VERİ ANALİZ RAPORU")
        print("=" * 60)
        
        # Temel istatistikler
        print("1. Temel İstatistikler:")
        print(f"   - Örnek sayısı: {len(df)}")
        print(f"   - Benzersiz şirketler: {df['company'].nunique()}")
        print(f"   - Tarih aralığı: {df['date'].min()} - {df['date'].max()}")
        
        # Skor dağılımı
        print("\n2. Skor Dağılımı:")
        score_stats = df['sentiment_score'].describe()
        print(f"   - Ortalama: {score_stats['mean']:.2f}")
        print(f"   - Medyan: {df['sentiment_score'].median():.2f}")
        print(f"   - Standart Sapma: {score_stats['std']:.2f}")
        print(f"   - Min: {score_stats['min']:.1f}")
        print(f"   - Max: {score_stats['max']:.1f}")
        
        # Metin uzunlukları
        df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
        print("\n3. Metin Uzunlukları:")
        length_stats = df['text_length'].describe()
        print(f"   - Ortalama kelime: {length_stats['mean']:.1f}")
        print(f"   - Min kelime: {length_stats['min']:.0f}")
        print(f"   - Max kelime: {length_stats['max']:.0f}")
        
        return df

def main():
    """Ana fonksiyon"""
    print("🚀 Finansal Duygu Analizi - Sentetik Veri Üretici")
    print("=" * 60)
    
    # Generator oluştur
    generator = FinancialDataGenerator()
    
    # Dataset oluştur
    df = generator.generate_dataset()
    
    # Dataset'i analiz et
    generator.analyze_dataset(df)
    
    # Dataset'i kaydet
    output_path = generator.save_dataset(df)
    
    print("\n" + "=" * 60)
    print("🎉 Sentetik veri üretimi başarıyla tamamlandı!")
    print(f"📁 Veri dosyası: {output_path}")
    
    # İlk 3 örneği göster
    print("\n📝 Örnek Metinler:")
    for i in range(min(3, len(df))):
        print(f"\n{i+1}. Skor: {df.iloc[i]['sentiment_score']:5.1f} - {df.iloc[i]['category']}")
        print(f"   Metin: {df.iloc[i]['text']}")

if __name__ == "__main__":
    main()