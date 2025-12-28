# 📉 Financial Sentiment Analysis (CNN-BiLSTM)

Türkçe finansal metinlerin duygu yoğunluğunu **-10 (Çok Olumsuz)** ile **+10 (Çok Olumlu)** aralığında tahminleyen, CNN ve BiLSTM katmanlarını birleştiren hibrit derin öğrenme projesi.

## ⚙️ Model Mimarisi

Model, metinlerdeki yerel n-gram özelliklerini ve uzun vadeli bağlamsal ilişkileri yakalamak için hibrit bir yapı kullanır:

1.  **Input & Embedding:** Tokenize edilmiş metin girişleri (Max len: 150).
2.  **CNN (Conv1D + MaxPool):** Metindeki yerel kalıpların (feature extraction) çıkarılması.
3.  **BiLSTM (Bidirectional LSTM):** Geçmiş ve gelecek bağlamının (sequential learning) öğrenilmesi.
4.  **Global Average Pooling:** Model karmaşıklığını azaltma ve özetleme.
5.  **Dense Output:** Linear aktivasyon fonksiyonu ile regresyon çıktısı (-10, +10).

## 📊 Performans (Test Seti)

* **R² Score:** 0.82
* **MAE (Ortalama Mutlak Hata):** 1.33
* **RMSE:** 1.65

## 🚀 Kurulum

```bash
git clone [https://github.com/hsamiarici/Financial_Duygu_Analizi.git](https://github.com/hsamiarici/Financial_Duygu_Analizi.git)
cd Financial_Duygu_Analizi
pip install -r requirements.txt