import gradio as gr
import tensorflow as tf
import pickle
import numpy as np
import re
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. AYARLAR
MODEL_PATH = 'models/model.h5'
TOKENIZER_PATH = 'models/tokenizer.pickle'
MAX_SEQUENCE_LENGTH = 150

# 2. MODEL VE TOKENIZER YÜKLEME
print("Sistem yükleniyor...")

try:
    # Modeli compile=False ile yükle (Hata riskini azaltır)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("✅ Model ve Tokenizer hazır!")
except Exception as e:
    print(f"❌ Hata oluştu: {e}")
    model = None
    tokenizer = None

# 3. YARDIMCI FONKSİYONLAR
def preprocess_text(text):
    """Metni temizle ve modele hazır hale getir"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded

def analyze_sentiment(text):
    """Gradio için tahmin fonksiyonu"""
    if model is None or tokenizer is None:
        return "Model yüklenemedi!", 0.0

    if not text.strip():
        return "Lütfen bir metin girin.", 0.0

    # Tahmin
    processed = preprocess_text(text)
    prediction = model.predict(processed, verbose=0)
    score = float(prediction[0][0])
    
    # Skoru -10 ile +10 arasına sabitle
    score = max(min(score, 10.0), -10.0)
    
    # Yorumlama
    if score >= 6:
        label = "Çok Olumlu 🚀"
    elif score >= 2:
        label = "Olumlu 📈"
    elif score >= -2:
        label = "Nötr 😐"
    elif score >= -6:
        label = "Olumsuz 📉"
    else:
        label = "Çok Olumsuz 💥"
        
    return label, score

# 4. GRADIO ARAYÜZÜ
# Tema ve bileşenler
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📊 Finansal Duygu Analizi Botu
        Bu model, Türkçe finansal metinleri analiz ederek **-10 (Çok Olumsuz)** ile **+10 (Çok Olumlu)** arasında puanlar.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Finansal Haber/Metin Giriniz", 
                placeholder="Örn: Şirket bu çeyrekte rekor kâr açıkladı...",
                lines=5
            )
            analyze_btn = gr.Button("Analiz Et", variant="primary")
            
            gr.Examples(
                examples=[
                    ["Merkez bankası faizleri sabit tuttu, piyasa sakin."],
                    ["Şirket iflas erteleme istedi, hisseler taban yaptı."],
                    ["İhracat rakamları beklentilerin çok üzerinde geldi."]
                ],
                inputs=input_text
            )

        with gr.Column():
            output_label = gr.Label(label="Duygu Durumu")
            output_score = gr.Number(label="Duygu Skoru (-10 ile +10)")

    # Buton aksiyonu
    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=input_text,
        outputs=[output_label, output_score]
    )

# Başlat
if __name__ == "__main__":
    demo.launch()