# 📈 Financial Sentiment Analysis (CNN–BiLSTM) + Streamlit / Gradio Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

Bu proje, Türkçe finansal metinlerden (haberler, piyasa yorumları, KAP bildirimleri) **duygu yoğunluğunu** -10 (çok olumsuz) ile +10 (çok olumlu) arasında **sürekli bir skor** olarak tahmin etmeyi amaçlayan bir yapay zeka uygulamasıdır.

Klasik *sınıflandırma* yaklaşımı yerine, **CNN (Convolutional Neural Networks)** ve **BiLSTM (Bidirectional LSTM)** katmanlarını birleştiren hibrit bir derin öğrenme mimarisi ile **regresyon tabanlı** duygu analizi gerçekleştirilmiştir.

---

## 📌 Özet (Ne Yapıldı?)

* **Veri Üretimi:** Finansal terminolojiye uygun, etiketli sentetik veri üreten özel bir modül geliştirildi (`src/data_generator.py`).
* **Ön İşleme:** Türkçe metin temizleme, karakter normalizasyonu (kâr → kar), tokenization ve sequence padding adımları uygulandı.
* **Model Mimarisi:**

  * Yerel kelime kalıplarını (n-gram) yakalamak için **CNN (Conv1D)**,
  * Bağlamsal ve zamansal ilişkileri öğrenmek için **BiLSTM** kullanıldı.
* **Arayüzler:**

  * **Gradio:** Hızlı demo ve sunum amaçlı web arayüzü.
  * **Streamlit:** Detaylı analiz, grafik ve görselleştirme paneli.
* **Performans:** Test seti üzerinde **R² = %82** açıklayıcılık skoru elde edildi.

---

## 🧠 Problem Tanımı ve Motivasyon

Finansal piyasalarda haber akışı fiyatları doğrudan etkiler; ancak her olumlu ya da olumsuz haberin etkisi aynı şiddette değildir.

* *“Şirket kâr açıkladı”* → Hafif olumlu (**+2.0**)
* *“Şirket tarihinin en yüksek kârını açıkladı ve temettü dağıtacak”* → Çok olumlu (**+9.0**)

Bu proje, metinleri basit sınıflar (Pozitif / Negatif) yerine **bir regresyon problemi** olarak ele alarak haberlerin **etki şiddetini** tahmin etmeyi hedefler.

---

## 📊 Veri Seti

Veri seti, proje kapsamında geliştirilen `src/data_generator.py` modülü ile üretilmiştir.

### 1. Ham Veri (`data/samples.csv`)

* Finansal terim sözlüğü (boğa, ayı, temettü, bilanço, tavan, taban vb.) kullanılarak oluşturulan sentetik cümleler
* **Hedef Değer:** -10.0 ile +10.0 arasında ondalıklı skor
* **Örnek Sayısı:** 2000+ (ölçeklenebilir)

### 2. İşlenmiş Veri (`data/processed/`)

* `processed_data.pickle`: Tokenize edilmiş, temizlenmiş ve padding uygulanmış eğitim / test setleri

---

## 🧩 Feature Engineering & Preprocessing

* **Text Cleaning:** URL, hashtag, kullanıcı adı ve noktalama işaretlerinin temizlenmesi (`src/utils.py`)
* **Normalization:** Türkçe karakterlerin sadeleştirilmesi (örn. *düşüş* → *dusus*) ile kelime kaybının azaltılması
* **Tokenization:** En sık kullanılan kelimelerin indekslenmesi (Vocabulary size ≈ 600)
* **Sequence Padding:** Tüm metinlerin sabit uzunluğa getirilmesi (Max length = 150)

---

## 🧪 Model Mimarisi (CNN–BiLSTM)

Model, metinlerdeki hem yerel kelime kalıplarını hem de uzun vadeli bağlamsal ilişkileri öğrenmek üzere tasarlanmıştır:

1. **Embedding Layer** – Kelimeleri yoğun vektör temsillerine dönüştürür
2. **Conv1D (CNN) Katmanları** – 3’lü ve 5’li n-gram filtreleri ile “rekor kâr”, “sert düşüş” gibi kalıpları yakalar
3. **BiLSTM Katmanı** – Metni çift yönlü okuyarak bağlam bütünlüğünü sağlar
4. **Global Average Pooling** – Özellikleri özetler ve overfitting riskini azaltır
5. **Dense Output Layer** – Linear aktivasyon ile -10 / +10 arası sürekli skor üretir

---

## ✅ Deneysel Sonuçlar (Test Seti)

| Metrik       | Değer    | Açıklama                           |
| ------------ | -------- | ---------------------------------- |
| **R² Score** | **0.82** | Model varyansın %82’sini açıklıyor |
| **MAE**      | **1.33** | Ortalama mutlak hata               |
| **RMSE**     | **1.65** | Karesel ortalama hata              |

> **Yorum:** Model, duygu yönünü yüksek doğrulukla yakalamakta; şiddet tahminlerinde ise insan algısına oldukça yakın sonuçlar üretmektedir.

---

## 💻 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın

```bash
git clone https://github.com/hsamiarici/Financial_Duygu_Analizi.git
cd Financial_Duygu_Analizi
```

### 2. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatın

**Sunum Modu (Gradio):**

```bash
python gradio_app.py
```

**Analiz Modu (Streamlit):**

```bash
streamlit run app.py
```

---

## 🔄 Modeli Yeniden Eğitme (Opsiyonel)

```bash
# 1. Yeni veri üret
python src/data_generator.py

# 2. Veriyi işle ve tokenize et
python notebooks/data_preprocessing.py

# 3. Modeli eğit
python notebooks/model_training.py
```

---

## 🔍 Sınırlılıklar ve Gelecek Çalışmalar

* **Veri Çeşitliliği:** Mevcut sürümde sentetik veri kullanılmaktadır. Gelecekte KAP, haber siteleri veya sosyal medya API’leri ile gerçek veri entegrasyonu planlanmaktadır.
* **Model Geliştirme:** BERT / FinBERT gibi transformer tabanlı modellerle performans karşılaştırmaları yapılabilir.

---

## 📁 Proje Yapısı

```text
Financial_Duygu_Analizi/
├── app.py                 # Streamlit arayüzü
├── gradio_app.py          # Gradio demo uygulaması
├── requirements.txt       # Bağımlılıklar
├── config.yaml            # Model ve veri ayarları
├── src/
│   ├── data_generator.py  # Veri üretim modülü
│   ├── model.py           # Model mimarisi
│   └── utils.py           # Yardımcı fonksiyonlar
├── notebooks/
│   ├── data_preprocessing.py
│   └── model_training.py
├── models/                # Eğitilmiş model ve tokenizer
└── data/                  # Veri setleri
```
