# 📈 Financial Sentiment Analysis (CNN-BiLSTM) + Streamlit / Gradio Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Gradio](https://img.shields.io/badge/Gradio-Demo-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

Bu proje, Türkçe finansal metinlerden (haberler, kullanıcı yorumları, KAP bildirimleri) **duygu yoğunluğunu** -10 ile +10 arasında sürekli bir skor olarak tahmin etmeyi amaçlamaktadır.

Klasik *olumlu / olumsuz* sınıflandırma yaklaşımı yerine, **CNN (Convolutional Neural Networks)** ve **BiLSTM (Bidirectional LSTM)** katmanlarını birleştiren hibrit bir derin öğrenme mimarisi ile **regresyon tabanlı** bir duygu skorlama yapılmıştır.

---

## 📌 Özet (Ne Yapıldı?)

- **Veri Üretimi:** Finansal terminolojiye uygun, etiketli sentetik veri üreten özel bir modül geliştirildi (`src/data_generator.py`).
- **Ön İşleme:** Türkçe metin temizleme, tokenization ve sequence padding adımları uygulandı.
- **Model Mimarisi:**  
  - Yerel kelime kalıplarını yakalamak için **CNN**,  
  - Bağlamsal ve zamansal ilişkileri öğrenmek için **BiLSTM** kullanıldı.
- **Arayüzler:**  
  - **Gradio:** Hızlı demo ve sunum amaçlı,  
  - **Streamlit:** Detaylı analiz ve görselleştirme paneli.
- **Performans:** Test seti üzerinde **%82 R² skoru** elde edildi.

---

## 🧠 Problem Tanımı ve Motivasyon

Finansal piyasalarda haber akışı fiyatları doğrudan etkiler; ancak her olumlu ya da olumsuz haberin etkisi aynı şiddette değildir.

- *“Şirket kâr açıkladı”* → Hafif olumlu (+2)  
- *“Şirket tarihinin en yüksek kârını açıkladı ve temettü dağıtacak”* → Çok olumlu (+9)

Bu proje, metinleri basit sınıflar yerine **bir regresyon problemi** olarak ele alarak haberlerin **etki şiddetini** tahmin etmeyi hedefler.

---

## 📊 Veri Seti

Veri seti, proje kapsamında geliştirilen `src/data_generator.py` modülü ile üretilmiştir.

### 1. Ham Veri (`data/samples.csv`)
- Finansal terim sözlüğü (boğa, ayı, temettü, bilanço vb.) kullanılarak oluşturulan sentetik cümleler
- **Hedef Değer:** -10 (çok olumsuz) ile +10 (çok olumlu) arasında ondalıklı skor
- **Örnek Sayısı:** 2000+ (ölçeklenebilir)

### 2. İşlenmiş Veri (`data/processed/`)
- `processed_data.pickle`: Tokenize edilmiş ve padding uygulanmış eğitim/test setleri

---

## 🧩 Feature Engineering & Preprocessing

- **Text Cleaning:** URL, hashtag, kullanıcı adı ve noktalama işaretlerinin temizlenmesi (`src/utils.py`)
- **Tokenization:** En sık kullanılan kelimelerin indekslenmesi (Vocabulary size ≈ 600)
- **Sequence Padding:** Tüm metinlerin sabit uzunluğa getirilmesi (Max length = 150)

---

## 🧪 Model Mimarisi (CNN–BiLSTM)

Model, metinlerdeki hem yerel kelime kalıplarını hem de uzun vadeli bağlamsal ilişkileri öğrenmek üzere tasarlanmıştır.

1. **Embedding Layer:** Kelimeleri yoğun vektör temsillerine dönüştürür  
2. **Conv1D (CNN) Katmanları:**  
   - 3’lü ve 5’li n-gram filtreleri ile “rekor kâr”, “sert düşüş” gibi kalıpları yakalar  
3. **BiLSTM Katmanı:**  
   - Metni çift yönlü okuyarak bağlam bütünlüğünü sağlar  
4. **Global Average Pooling:**  
   - Özellikleri özetler ve overfitting riskini azaltır  
5. **Dense Output Layer:**  
   - Linear aktivasyon ile -10 / +10 arası sürekli skor üretir  

---

## ✅ Deneysel Sonuçlar (Test Seti)

| Metrik | Değer | Açıklama |
|------|------|---------|
| **R² Score** | **0.82** | Model varyansın %82’sini açıklıyor |
| **MAE** | **1.33** | Ortalama mutlak hata |
| **RMSE** | **1.65** | Karesel ortalama hata |

> Model, duygu yönünü yüksek doğrulukla yakalamakta; şiddet tahminlerinde ise sınırlı sapmalar göstermektedir.

---
##  Modeli Sıfırdan Eğitme (Opsiyonel)
python src/data_generator.py
python notebooks/data_preprocessing.py
python notebooks/model_training.py


## 🔍 Sınırlılıklar ve Gelecek Çalışmalar

Veri Çeşitliliği: Şu anda sentetik veri kullanılmaktadır. Gelecekte KAP veya sosyal medya API’leri ile gerçek veri entegrasyonu planlanmaktadır.

Model Geliştirme: BERT / FinBERT gibi transformer tabanlı modellerle performans karşılaştırmaları yapılabilir.


## 📁 Proje Yapısı

Financial_Duygu_Analizi/
├── app.py
├── gradio_app.py
├── requirements.txt
├── config.yaml
├── src/
│   ├── data_generator.py
│   ├── model.py
│   └── utils.py
├── notebooks/
│   ├── data_preprocessing.py
│   └── model_training.py
├── models/
└── data/

## 🚀 Kurulum

```bash
git clone https://github.com/hsamiarici/Financial_Duygu_Analizi.git
cd Financial_Duygu_Analizi
pip install -r requirements.txt


