# src/utils.py
import yaml
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords

def load_config(config_path="config.yaml"):
    """Yapılandırma dosyasını yükle"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def clean_text(text):
    """Metni temizle"""
    if not isinstance(text, str):
        return ""
    
    # Küçük harfe çevir
    text = text.lower()
    
    # URL'leri kaldır
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Kullanıcı adlarını kaldır
    text = re.sub(r'@\w+', '', text)
    
    # Hashtag'lerden # işaretini kaldır
    text = re.sub(r'#', '', text)
    
    # Rakamları kaldır
    text = re.sub(r'\d+', '', text)
    
    # Noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_data(df, config):
    """Veriyi hazırla ve tokenize et"""
    # Metinleri temizle
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Train/val/test split
    X = df['cleaned_text'].values
    y = df['sentiment_score'].values
    
    # Stratified split (skor aralıklarına göre)
    bins = np.linspace(-10, 10, 21)
    y_binned = np.digitize(y, bins)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config['data']['val_size'] + config['data']['test_size'],
        random_state=42, stratify=y_binned
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=config['data']['test_size']/(config['data']['val_size'] + config['data']['test_size']),
        random_state=42, stratify=np.digitize(y_temp, bins)
    )
    
    # Tokenizer
    tokenizer = Tokenizer(num_words=config['data']['vocab_size'], oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    # Sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Padding
    max_len = config['data']['max_sequence_length']
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    
    # Tokenizer'ı kaydet
    with open(config['paths']['tokenizer'], 'wb') as f:
        pickle.dump(tokenizer, f)
    
    return {
        'X_train': X_train_pad,
        'X_val': X_val_pad,
        'X_test': X_test_pad,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'tokenizer': tokenizer,
        'vocab_size': min(config['data']['vocab_size'], len(tokenizer.word_index) + 1)
    }