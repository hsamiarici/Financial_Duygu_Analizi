import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. AYARLAR VE YAPILANDIRMA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Finansal Duygu Analizi",
    page_icon="📈",
    layout="wide"
)

# Arayüz için özel CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 2rem; }
    .stTextArea textarea { font-size: 1.1rem; }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Dosya yolları
MODEL_PATH = 'models/model.h5'
TOKENIZER_PATH = 'models/tokenizer.pickle'
MAX_SEQUENCE_LENGTH = 150  # Eğitimdeki max_len değeri ile aynı olmalı

# -----------------------------------------------------------------------------
# 2. MODEL VE TOKENIZER YÜKLEME
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    """Model ve tokenizer'ı önbelleğe alarak yükler"""
    resources = {'model': None, 'tokenizer': None}
    
    # Model Yükleme
    if os.path.exists(MODEL_PATH):
        try:
            # compile=False ile yüklemek daha güvenli ve hızlıdır (sadece tahmin yapacağız)
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            resources['model'] = model
        except Exception as e:
            st.error(f"❌ Model yüklenirken hata oluştu: {e}")
    else:
        st.error(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")

    # Tokenizer Yükleme
    if os.path.exists(TOKENIZER_PATH):
        try:
            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer = pickle.load(f)
            resources['tokenizer'] = tokenizer
        except Exception as e:
            st.error(f"❌ Tokenizer yüklenirken hata oluştu: {e}")
    else:
        st.error(f"❌ Tokenizer dosyası bulunamadı: {TOKENIZER_PATH}")
        
    return resources

# -----------------------------------------------------------------------------
# 3. YARDIMCI FONKSİYONLAR
# -----------------------------------------------------------------------------
def preprocess_text(text, tokenizer):
    """Metni modelin anlayacağı formata çevirir"""
    if not text or not tokenizer:
        return None
    
    # Temizlik
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize ve Padding
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded

def create_gauge_chart(score):
    """Skor göstergesi (Gauge Chart) oluşturur"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Duygu Skoru", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [-10, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "white", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-10, -6], 'color': '#ff4b4b'},  # Kırmızı (Çok Olumsuz)
                {'range': [-6, -2], 'color': '#ff9f43'},   # Turuncu (Olumsuz)
                {'range': [-2, 2], 'color': '#feca57'},    # Sarı (Nötr)
                {'range': [2, 6], 'color': '#48dbfb'},     # Mavi (Olumlu)
                {'range': [6, 10], 'color': '#1dd1a1'}     # Yeşil (Çok Olumlu)
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# -----------------------------------------------------------------------------
# 4. ANA UYGULAMA
# -----------------------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">📊 Finansal Metin Duygu Analizi</h1>', unsafe_allow_html=True)
    
    # Sidebar - Durum Bilgisi
    with st.sidebar:
        st.header("Sistem Durumu")
        resources = load_resources()
        
        if resources['model'] and resources['tokenizer']:
            st.success("✅ Sistem Hazır")
            st.info("Model: CNN-BiLSTM\nVeri: Finansal Haberler")
        else:
            st.error("❌ Sistem Yüklenemedi")
            st.stop()
            
        st.markdown("---")
        st.markdown("### Hakkında")
        st.write("Bu model finansal metinleri -10 (Çok Olumsuz) ile +10 (Çok Olumlu) arasında puanlar.")

    # Ana Ekran
    col_input, col_result = st.columns([1.5, 1])
    
    with col_input:
        st.subheader("📝 Metin Girişi")
        user_input = st.text_area(
            "Analiz edilecek finansal haberi veya cümleyi girin:",
            height=200,
            placeholder="Örn: Şirketin bu çeyrekteki kârı beklentilerin çok üzerinde geldi, hisseler tavan yaptı."
        )
        
        analyze_button = st.button("Analiz Et 🚀", type="primary", use_container_width=True)
        
        # Örnek Butonları
        st.markdown("#### veya örnek seçin:")
        examples = [
            "Şirket rekor kâr açıkladı, yatırımcılar çok mutlu.",
            "Borsa günü sert düşüşle kapattı, piyasada panik var.",
            "Merkez bankası faiz kararını açıkladı, piyasa tepkisiz."
        ]
        
        cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            if cols[i].button(f"Örnek {i+1}", use_container_width=True):
                st.session_state.temp_input = ex
                # Not: Streamlit buton mantığı gereği metni text_area'ya aktarmak için 
                # rerun gerekebilir ama basitlik adına kullanıcı kopyalayıp yapıştırabilir.
                st.info(f"Seçilen: {ex}")

    # Analiz İşlemi
    if analyze_button and user_input:
        with st.spinner("Yapay zeka analiz yapıyor..."):
            try:
                # Preprocess
                processed_data = preprocess_text(user_input, resources['tokenizer'])
                
                # Predict
                prediction = resources['model'].predict(processed_data, verbose=0)
                score = float(prediction[0][0])
                
                # Skoru sınırla (-10 ile +10 arası)
                score = max(min(score, 10.0), -10.0)
                
                # Sonucu Session State'e kaydet (grafik yenilenince kaybolmasın diye)
                st.session_state.last_score = score
                st.session_state.last_text = user_input
                
            except Exception as e:
                st.error(f"Analiz hatası: {e}")

    # Sonuç Ekranı
    with col_result:
        if 'last_score' in st.session_state:
            score = st.session_state.last_score
            
            st.subheader("🎯 Sonuç")
            
            # Gösterge Grafiği
            fig = create_gauge_chart(score)
            st.plotly_chart(fig, use_container_width=True)
            
            # Yazılı Yorum
            if score >= 6:
                bg_color, text = "#1dd1a1", "Çok Olumlu"
            elif score >= 2:
                bg_color, text = "#48dbfb", "Olumlu"
            elif score >= -2:
                bg_color, text = "#feca57", "Nötr/Dengeli"
            elif score >= -6:
                bg_color, text = "#ff9f43", "Olumsuz"
            else:
                bg_color, text = "#ff4b4b", "Çok Olumsuz"
            
            st.markdown(f"""
            <div class="prediction-box" style="background-color: {bg_color};">
                <h2 style="margin:0; color:white; text-shadow: 1px 1px 2px black;">{text}</h2>
                <h3 style="margin:0; color:white;">{score:.2f} / 10</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👈 Analiz sonucunu görmek için soldan metin girip butona basın.")

if __name__ == "__main__":
    main()