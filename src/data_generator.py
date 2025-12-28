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
        
        # --- GÜNCELLENMİŞ SÖZLÜK (Normalize edilmiş) ---
        # utils.py'deki clean_text fonksiyonu ile tam uyumlu olması için 
        # Türkçe karakterlerden arındırılmış kelimeler kullanıyoruz.
        self.financial_terms = {
            'positive': [
                'yukselis', 'artis', 'kazanc', 'kar', 'buyume', 'olumlu',
                'guclu', 'saglam', 'basarili', 'iyi', 'rekor', 'muthis',
                'harika', 'pozitif', 'gelisme', 'ilerleme', 'fayda', 'getiri',
                'karli', 'verimli', 'istikrarli', 'guvenli', 'cazip', 'parlak',
                'tavan', 'zirve', 'patlama', 'cosku', 'ralli', 'firsat', 'prim'
            ],
            'negative': [
                'dusus', 'kayip', 'zarar', 'basarisiz', 'kotu', 'zayif',
                'olumsuz', 'dusuk', 'kriz', 'risk', 'korku', 'panik',
                'kaygi', 'endise', 'cokus', 'negatif', 'tehlikeli', 
                'istikrarsiz', 'belirsiz', 'tedirgin', 'baski',
                'taban', 'iflas', 'borc', 'temerrut', 'satis', 'cikis', 'dip'
            ],
            'companies': [
                'THYAO', 'AKBNK', 'GARAN', 'ISCTR', 'YKBNK', 'ASELS',
                'TAVHL', 'BIMAS', 'MGROS', 'SAHOL', 'KCHOL', 'SASA',
                'PETKM', 'TUPRS', 'FROTO', 'TCELL', 'EREGL', 'TOASO',
                'ARCLK', 'HEKTS', 'ENKAI', 'KORDS', 'VAKBN', 'GUBRF',
                'ODAS', 'KOZAL', 'KARDM', 'VESTL', 'SISE', 'PGSUS'
            ],
            'sectors': [
                'banka', 'havayolu', 'savunma', 'perakende', 'otomotiv',
                'enerji', 'kimya', 'telekom', 'celik', 'gayrimenkul',
                'teknoloji', 'saglik', 'ulasim', 'gida', 'insaat',
                'turizm', 'madencilik', 'tekstil'
            ],
            'verbs': [
                'acikladi', 'bildirdi', 'duyurdu', 'ilan etti', 'paylasti',
                'raporladi', 'sunuldu', 'belirtildi', 'ifade edildi',
                'aciklandi', 'yayinlandi', 'iletildi', 'bildirildi',
                'gerceklesti', 'tamamlandi'
            ],
            'nouns': [
                'kar', 'ciro', 'satis', 'buyume', 'performans', 'sonuc',
                'rapor', 'veri', 'istatistik', 'analiz', 'tahmin', 'beklenti',
                'projeksiyon', 'ongoru', 'degerlendirme', 'inceleme',
                'bilanco', 'temettu', 'gelir', 'hedef', 'butce'
            ]
        }
        
        self.templates = self._create_templates()
    
    def _create_templates(self):
        """Duygu skorlarına göre template'ler oluştur"""
        return {
            'very_positive': [
                "{company} hissesi {positive} bir {noun} {verb}! Yatirimcilar mutlu.",
                "Sirketin {positive} {noun} aciklamasi piyasayi hareketlendirdi.",
                "{company} icin {positive} haberler geliyor, fiyatlar tirmaniyor.",
                "Analistler {company} hissesine guclu alim onerisi verdi.",
                "{sector} sektorunde {positive} gelismeler yasaniyor.",
                "{company} {positive} {noun} rakamlariyla dikkat cekti.",
                "Yatirimcilar {company} hissesinde {positive} hareket bekliyor.",
                "{company} {verb} {positive} bir {noun} performansi sergiledi."
            ],
            'positive': [
                "{company} hissesinde {positive} yonde hareketler gozleniyor.",
                "Sirketin {noun} performansi {positive} olarak degerlendiriliyor.",
                "{company} icin {positive} sinyaller aliniyor.",
                "{sector} sektoru {positive} bir seyir izliyor.",
                "{company} {verb} {positive} {noun} verileri.",
                "Piyasada {company} hissesine yonelik {positive} hava hakim.",
                "{company} hissesi {positive} bir trend icinde."
            ],
            'neutral': [
                "{company} hissesi normal seyirde ilerliyor.",
                "Sirket beklentileri karsiladi, piyasa tepkisiz.",
                "{company} hissesinde onemli bir hareket yok.",
                "{sector} sektorunde dengeli bir seyir hakim.",
                "{company} {verb} beklenen {noun} rakamlarini.",
                "Piyasa {company} hissesini izlemeye devam ediyor.",
                "{company} hissesi teknik analizde notr bolgede."
            ],
            'negative': [
                "{company} hissesinde {negative} yonde gelismeler var.",
                "Sirketin {noun} performansi {negative} olarak degerlendirildi.",
                "{company} icin {negative} riskler goruluyor.",
                "{sector} sektorunde {negative} hava hakim.",
                "{company} {verb} {negative} {noun} verileri.",
                "Piyasada {company} hissesine yonelik {negative} beklentiler var.",
                "{company} hissesi {negative} bir trende girdi."
            ],
            'very_negative': [
                "{company} hissesi {negative} bir {noun} {verb}! Yatirimcilar endiseli.",
                "Sirketten gelen {negative} haberler piyasayi sarsti.",
                "{company} hissesi icin alarm zilleri caliyor.",
                "{sector} sektorunde kriz sinyalleri artiyor.",
                "{company} {negative} {noun} rakamlariyla sok etti.",
                "Yatirimcilar {company} hissesinden hizla cikis yapiyor.",
                "{company} {verb} {negative} bir {noun} performansi.",
                "{company} hissesi taban oldu, {negative} haberler etkili."
            ]
        }
    
    def _get_sentiment_category(self, score):
        """Skora göre duygu kategorisi belirle"""
        if score >= 7: return 'very_positive'
        elif score >= 3: return 'positive'
        elif score >= -2: return 'neutral'
        elif score >= -6: return 'negative'
        else: return 'very_negative'
    
    def _add_variations(self, text, score):
        """Metne çeşitlilik ekle (hashtag vb.)"""
        variations = []
        # Hashtag ekle (%40 ihtimal)
        if random.random() < 0.4:
            hashtags = ['#borsa', '#yatirim', '#hisse', '#finans', '#ekonomi', 
                       '#bist', '#piyasa', '#analiz', '#trade', '#para']
            text += " " + random.choice(hashtags)
            variations.append("hashtag")
        return text, variations
    
    def generate_text(self, sentiment_score):
        """Duygu skoruna göre metin üret"""
        category = self._get_sentiment_category(sentiment_score)
        template = random.choice(self.templates[category])
        
        replacements = {
            'company': random.choice(self.financial_terms['companies']),
            'sector': random.choice(self.financial_terms['sectors']),
            'verb': random.choice(self.financial_terms['verbs']),
            'noun': random.choice(self.financial_terms['nouns']),
            'positive': random.choice(self.financial_terms['positive']),
            'negative': random.choice(self.financial_terms['negative'])
        }
        
        text = template.format(**replacements)
        text, variations = self._add_variations(text, sentiment_score)
        return text
    
    def generate_dataset(self, num_samples=None):
        """Sentetik dataset oluştur"""
        if num_samples is None:
            num_samples = self.config['data']['num_samples']
        
        print(f"📊 {num_samples} adet sentetik finansal veri üretiliyor...")
        print("=" * 60)
        
        data = []
        progress_step = max(1, num_samples // 10)
        
        for i in range(num_samples):
            # Normal dağılım ile skor üretimi
            raw_score = np.random.normal(0, 4)
            sentiment_score = np.clip(raw_score, -10, 10)
            sentiment_score = round(sentiment_score, 1)
            
            text = self.generate_text(sentiment_score)
            
            # Rastgele tarih
            start_date = datetime(2022, 1, 1)
            random_date = start_date + timedelta(days=random.randint(0, 729))
            
            # Kategori
            if sentiment_score >= 7: category = "Çok Olumlu"
            elif sentiment_score >= 3: category = "Olumlu"
            elif sentiment_score >= -2: category = "Nötr"
            elif sentiment_score >= -6: category = "Olumsuz"
            else: category = "Çok Olumsuz"
            
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
            
            if (i + 1) % progress_step == 0:
                progress = (i + 1) / num_samples * 100
                print(f"  ██████████████████ {progress:.0f}% tamamlandı ({i + 1}/{num_samples})")
        
        df = pd.DataFrame(data)
        
        print("\n" + "=" * 60)
        print("✅ Veri üretimi tamamlandı!")
        print(f"\n📈 İstatistikler:")
        print(f"   Toplam örnek: {len(df)}")
        print(f"   Skor aralığı: {df['sentiment_score'].min():.1f} - {df['sentiment_score'].max():.1f}")
        
        return df
    
    def save_dataset(self, df, filepath=None):
        """Dataset'i kaydet"""
        if filepath is None:
            filepath = self.config['paths']['raw_data']
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # UTF-8-SIG kullanarak kaydet (Excel'de Türkçe karakter sorunu olmasın diye, 
        # gerçi içerik normalize ama başlıklar için iyi olur)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n💾 Veri kaydedildi: {filepath}")
        return filepath
    
    def analyze_dataset(self, df):
        """Basit analiz"""
        print("\n📊 VERİ ANALİZ RAPORU")
        print("-" * 20)
        print(f"Örnek sayısı: {len(df)}")
        return df

def main():
    print("🚀 Finansal Duygu Analizi - Sentetik Veri Üretici")
    print("=" * 60)
    generator = FinancialDataGenerator()
    df = generator.generate_dataset()
    generator.analyze_dataset(df)
    output_path = generator.save_dataset(df)
    print(f"\n✅ İşlem tamamlandı. Dosya: {output_path}")

if __name__ == "__main__":
    main()